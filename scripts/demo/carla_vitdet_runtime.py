#!/usr/bin/env python3
import copy
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from time import perf_counter

import supervision as sv
from datasets.kitti_tracking import make_coco_transforms
from models.vitdet import ViTDet
from scripts.evaluate.vitdet_kitti_mask_bytetrack_dynmask_safe import (
    SlidingWindowHeatmap,
    detach_results_for_mask,
    evaluate_safety_environment,
    get_region_mask_from_results,
    merge_mask_indices,
    results_to_supervision_detections,
    tracker_predict_detections,
    update_track_metadata,
)
from utils.config import load_config
from utils.misc import get_pytorch_device


class _CameraState:
    def __init__(self, tracker, heatmap_state):
        self.tracker = tracker
        self.track_metadata = {}
        self.prev_results_for_mask = None
        self.safe_tracker_only = False
        self.prev_detection_boxes = {}
        self.step = 0
        self.heatmap_state = heatmap_state
        self.heatmap_mask_cache = None


class VitDetCarlaRuntime:
    """
    将 Kitti 评测脚本中的动态 mask / 热图 / 安全追踪逻辑封装成
    可在 CARLA 每帧调用的检测器。
    """

    def __init__(
        self,
        config_path="configs/evaluate/vitdet_kitti/kitti_mask_0.yml",
        device=None,
    ):
        raw_cfg = load_config(Path(config_path), to_container=False)
        if "_name" not in raw_cfg:
            raw_cfg["_name"] = Path(config_path).stem
        self.config = OmegaConf.to_container(raw_cfg, resolve=True)
        device = device or get_pytorch_device()
        self.device = torch.device(device) if isinstance(device, str) else device

        cfg = copy.deepcopy(self.config)
        n_classes = cfg.get("nb_classes", cfg["model"].get("classes", 1))
        cfg["model"]["classes"] = n_classes
        cfg["model"]["mask"] = True

        self.model = ViTDet(**cfg["model"])
        msg = self.model.load_state_dict(torch.load(cfg["weights"]), strict=False)
        print(msg)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.region_size = int(cfg.get("region_size", 16))
        self.period = int(cfg.get("period", 1))
        self.conf = float(cfg.get("conf", 0.5))
        self.target_sparsity = float(cfg.get("sparsity", 1.0))
        self.mask_margin = int(cfg.get("margin", 0))
        self.tracker_cfg = cfg.get("tracker", {})
        self.safety_cfg = cfg.get("safety", {})

        img_shape = cfg["model"]["input_shape"][-2:]
        self.total_region_tokens = (img_shape[0] // self.region_size) * (
            img_shape[1] // self.region_size
        )

        self.heatmap_window = int(cfg.get("dynamic_heatmap_window", 30))
        self.heatmap_min_frames = int(
            cfg.get("dynamic_heatmap_min_frames", min(30, self.heatmap_window))
        )
        self.heatmap_min_activity = float(cfg.get("dynamic_heatmap_min_activity", 0.0))
        self.heatmap_score_threshold = float(
            cfg.get("dynamic_heatmap_score_threshold", 0.75)
        )
        self.heatmap_keep_ratio = max(0.0, min(1.0, 1 - self.target_sparsity))

        # 与评测数据一致的预处理：resize 到 (370, 672) 再 padding 到 672x672。
        self.transform = make_coco_transforms("val", (672, 370))
        self.resize_hw = (370, 672)

        self.states = {}
        # 使用自定义绘制以兼容旧版 supervision（避免 BoxAnnotator 参数兼容性问题）

    def _build_tracker(self):
        return sv.ByteTrack(
            track_activation_threshold=self.tracker_cfg.get(
                "track_activation_threshold", 0.5
            ),
            lost_track_buffer=self.tracker_cfg.get("lost_track_buffer", 30),
            minimum_matching_threshold=self.tracker_cfg.get(
                "minimum_matching_threshold", 0.8
            ),
            frame_rate=self.tracker_cfg.get("frame_rate", 30),
            minimum_consecutive_frames=self.tracker_cfg.get(
                "minimum_consecutive_frames", 1
            ),
        )

    def _new_state(self):
        heatmap_state = SlidingWindowHeatmap(
            image_shape=self.resize_hw[::-1],  # (H, W)
            region_size=self.region_size,
            window_size=self.heatmap_window,
            score_threshold=self.heatmap_score_threshold,
        )
        return _CameraState(self._build_tracker(), heatmap_state)

    def _get_state(self, camera_id):
        if camera_id not in self.states:
            self.states[camera_id] = self._new_state()
        return self.states[camera_id]

    def reset_camera(self, camera_id=None):
        if camera_id is None:
            self.states = {}
        else:
            self.states[camera_id] = self._new_state()

    def _prepare_tensor(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor, _ = self.transform(pil_img, {})
        return tensor

    def _rescale_boxes(self, boxes, orig_size):
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        h_resized, w_resized = self.resize_hw
        orig_h, orig_w = orig_size
        boxes_t[:, 0::2] = boxes_t[:, 0::2].clamp(0, w_resized) * (
            float(orig_w) / float(w_resized)
        )
        boxes_t[:, 1::2] = boxes_t[:, 1::2].clamp(0, h_resized) * (
            float(orig_h) / float(h_resized)
        )
        return boxes_t.cpu().numpy()

    def _annotate(self, frame_bgr, boxes, scores, labels, tracker_ids=None):
        if boxes is None or len(boxes) == 0:
            return frame_bgr
        vis = frame_bgr.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(np.int32).tolist()
            color = (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cls = int(labels[i]) if labels is not None and len(labels) > i else 0
            score = float(scores[i]) if scores is not None and len(scores) > i else 0.0
            text = f"c{cls}:{score:.2f}"
            if tracker_ids is not None and len(tracker_ids) > i:
                tid = int(tracker_ids[i])
                text = f"id{tid}-" + text
            cv2.putText(
                vis,
                text,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return vis

    def infer(self, camera_id, frame_bgr):
        """
        对单帧进行检测并返回绘制好的图像和检测结果。
        """
        if frame_bgr is None:
            return None
        system_start = perf_counter()
        model_latency_ms = 0.0
        tracker_latency_ms = 0.0
        state = self._get_state(camera_id)
        img_tensor = self._prepare_tensor(frame_bgr)
        orig_h, orig_w = frame_bgr.shape[:2]
        batch = img_tensor.unsqueeze(0).to(self.device)

        tracker = state.tracker
        invalid_track_id = tracker.external_id_counter.NO_ID
        has_active_tracks = any(
            track.is_activated and track.external_track_id != invalid_track_id
            for track in getattr(tracker, "tracked_tracks", [])
        )
        if not has_active_tracks:
            state.safe_tracker_only = False

        is_key_frame = (state.step % self.period) == 0
        run_model = True
        if is_key_frame:
            state.safe_tracker_only = False
        elif state.safe_tracker_only and has_active_tracks:
            run_model = False

        mask_index_tensor = None
        if run_model and not is_key_frame:
            dynamic_mask = get_region_mask_from_results(
                state.prev_results_for_mask,
                image_shape=self.resize_hw[::-1],
                conf_threshold=self.conf,
                region_size=self.region_size,
                margin=self.mask_margin,
            )
            mask_candidates = []
            if dynamic_mask is not None:
                mask_candidates.append(dynamic_mask)
            if state.heatmap_mask_cache is not None:
                mask_candidates.append(state.heatmap_mask_cache)
            if mask_candidates:
                mask_index_cpu = merge_mask_indices(*mask_candidates)
                if mask_index_cpu is not None:
                    mask_index_tensor = mask_index_cpu.to(self.device)

        if run_model:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            model_start = perf_counter()
            with torch.inference_mode():
                results, _ = self.model(batch, mask_index_tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            model_latency_ms = (perf_counter() - model_start) * 1000
            detections = results_to_supervision_detections(results[0])

            safety_enabled = self.safety_cfg.get("enabled", True)
            predicted_eval = None
            if safety_enabled and has_active_tracks:
                try:
                    tracker_snapshot = copy.deepcopy(tracker)
                    metadata_snapshot = copy.deepcopy(state.track_metadata)
                    predicted_eval = tracker_predict_detections(
                        tracker_snapshot, metadata_snapshot
                    )
                except Exception:
                    predicted_eval = None

            tracker_start = perf_counter()
            tracked = tracker.update_with_detections(detections)
            tracker_latency_ms = (perf_counter() - tracker_start) * 1000
            update_track_metadata(tracked, state.track_metadata)
            state.prev_results_for_mask = detach_results_for_mask(results)
            if state.heatmap_state is not None:
                frame_results = results[0]
                state.heatmap_state.update(
                    frame_results["boxes"], frame_results.get("scores")
                )
                new_heatmap_mask = state.heatmap_state.build_mask_index(
                    keep_ratio=self.heatmap_keep_ratio,
                    min_ready_frames=self.heatmap_min_frames,
                    min_activity=self.heatmap_min_activity,
                )
                if new_heatmap_mask is not None:
                    state.heatmap_mask_cache = new_heatmap_mask.cpu()

            # 更新安全追踪状态
            if safety_enabled and predicted_eval is not None:
                prev_boxes_snapshot = {
                    tid: box.copy()
                    for tid, box in state.prev_detection_boxes.items()
                }
                is_safe, _ = evaluate_safety_environment(
                    predicted_eval,
                    detections,
                    prev_boxes_snapshot,
                    self.safety_cfg,
                )
                if is_safe and not state.safe_tracker_only:
                    state.safe_tracker_only = True

            if getattr(tracked, "tracker_id", None) is not None:
                for idx, tracker_id in enumerate(tracked.tracker_id):
                    if tracker_id == -1:
                        continue
                    state.prev_detection_boxes[int(tracker_id)] = tracked.xyxy[idx].copy()

            boxes = tracked.xyxy
            scores = tracked.confidence
            labels = tracked.class_id
            tracker_ids = tracked.tracker_id if hasattr(tracked, "tracker_id") else None
        else:
            tracker_start = perf_counter()
            tracked = tracker_predict_detections(tracker, state.track_metadata)
            tracker_latency_ms = (perf_counter() - tracker_start) * 1000
            boxes = tracked.xyxy
            scores = tracked.confidence
            labels = tracked.class_id
            tracker_ids = tracked.tracker_id if hasattr(tracked, "tracker_id") else None

        boxes = self._rescale_boxes(boxes, (orig_h, orig_w))
        state.step += 1

        annotated = self._annotate(
            frame_bgr,
            boxes,
            scores if scores is not None else np.zeros(len(boxes)),
            labels if labels is not None else np.zeros(len(boxes), dtype=np.int64),
            tracker_ids=tracker_ids,
        )

        system_latency_ms = (perf_counter() - system_start) * 1000

        return {
            "annotated": annotated,
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "tracker_ids": tracker_ids,
            "model_latency_ms": model_latency_ms,
            "tracker_latency_ms": tracker_latency_ms,
            "system_latency_ms": system_latency_ms,
            "run_model": run_model,
        }


__all__ = ["VitDetCarlaRuntime"]

