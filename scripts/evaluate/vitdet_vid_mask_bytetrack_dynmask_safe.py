#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from datasets.vid import VIDResize, VID
from models.vitdet import ViTDet
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import dict_to_device, squeeze_dict, get_pytorch_device, compute_detection_recall
from torch import optim
from datetime import datetime
from time import perf_counter
from torch.utils.tensorboard import SummaryWriter
from detectron2.data.detection_utils import annotations_to_instances, BoxMode
from detectron2.utils.events import EventStorage
from utils.misc import tee_print
from backbones.base import dict_string
import torch.nn.functional as F
import copy
from util.lr_sched import LR_Scheduler
from torchprofile import profile_macs
import numpy as np
from utils.image import pad_to_size, rescale
import supervision as sv
from supervision.tracker.byte_tracker.single_object_track import STrack
import math


def collate_fn(batch):
    batch = list(zip(*batch))
    # batch[0] = nested_tensor_from_tensor_list(batch[0])
    # return tuple(batch)
    batch[0] = torch.stack(batch[0])
    return tuple(batch)

def get_original_mask(annotation, image_shape, region_size=16):
    mask = torch.zeros(image_shape)
    for i, bbox in enumerate(annotation['boxes']):
        x1, y1, x2, y2 = bbox
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
    weight = torch.ones(1, 1, region_size, region_size)
    y = F.conv2d(mask.unsqueeze(0).unsqueeze(0), weight, stride=region_size)
    mask_index = torch.nonzero(y.flatten() > 0).reshape(1, -1)
    sparsity = 1 - mask_index.shape[1] / y.numel()
    return mask_index, sparsity, y.numel()    


def topkmask(feature, sparsity=0.5):
    # import pdb; pdb.set_trace()
    feature_flat = feature.flatten(1)
    n_tokens = feature_flat.shape[1]
    left_tokens = int(n_tokens * sparsity)
    value, idx = torch.topk(feature_flat, left_tokens, dim=1, sorted=False)
    # min_value = value[:,-1].unsqueeze(-1).unsqueeze(-1).expand(-1, N, C) # min value to keep for each feature map
    # return (feature > min_value) + 0
    return idx

def get_region_mask_static(short_edge_length, max_size, region_size=16, region_sparsity=0.5):
    with open('heatmap_vid.npy', 'rb') as f:
        heat_map = np.load(f)
    heatmap = torch.tensor(heat_map/heat_map.max(), dtype=torch.float)

    short_edge = min(heatmap.shape[-2:])
    long_edge = max(heatmap.shape[-2:])
    scale = min(short_edge_length / short_edge, max_size / long_edge)
    heatmap = rescale(heatmap.unsqueeze(0).unsqueeze(0), scale)
    # heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), (370, 672), mode='bilinear')
    heatmap = pad_to_size(heatmap, (max_size, max_size))
    weight = torch.ones(1, 1, region_size, region_size)
    y = F.conv2d(heatmap, weight, stride=region_size)
    index = topkmask(y, sparsity=region_sparsity)
    # scores = y.flatten(1).softmax(1)
    return index


def detach_results_for_mask(results):
    if results is None:
        return None
    detached = []
    for result in results:
        detached.append(
            {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in result.items()
            }
        )
    return detached


def get_region_mask_from_results(
    prev_results,
    image_shape,
    conf_threshold=0.5,
    region_size=16,
    margin=0,
):
    if not prev_results:
        return None

    height, width = image_shape
    mask = torch.zeros((height, width), dtype=torch.float32)
    margin = max(int(margin), 0)
    has_box = False

    for result in prev_results:
        boxes = result["boxes"].detach().cpu()
        scores = result["scores"].detach().cpu()
        for idx, bbox in enumerate(boxes):
            if scores[idx] < conf_threshold:
                continue
            x1, y1, x2, y2 = bbox.tolist()
            x1 = max(0, int(np.floor(x1)) - margin)
            y1 = max(0, int(np.floor(y1)) - margin)
            x2 = min(width, int(np.ceil(x2)) + margin)
            y2 = min(height, int(np.ceil(y2)) + margin)
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1:y2, x1:x2] = 1
            has_box = True

    if not has_box:
        return None

    weight = torch.ones(1, 1, region_size, region_size)
    y = F.conv2d(mask.unsqueeze(0).unsqueeze(0), weight, stride=region_size)
    mask_index = torch.nonzero(y.flatten() > 0)
    if mask_index.numel() == 0:
        return None
    return mask_index.reshape(1, -1)


def results_to_supervision_detections(result):
    boxes = result["boxes"].detach()
    if boxes.numel() == 0:
        detections = sv.Detections.empty()
        detections.confidence = np.empty((0,), dtype=np.float32)
        detections.class_id = np.empty((0,), dtype=np.int64)
        return detections
    scores = result["scores"].detach()
    class_ids = result["labels"].detach()

    results = torch.cat((boxes, scores.unsqueeze(-1), class_ids.unsqueeze(-1)), dim=1).cpu().numpy()

    boxes = results[:, :4]
    scores = results[:, 4]
    class_ids = results[:, 5].astype(np.int64, copy=False)
    
    return sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=class_ids,
    )


def detections_to_result_dict(detections, device):
    if len(detections) == 0:
        boxes = torch.empty((0, 4), device=device)
        scores = torch.empty((0,), device=device)
        labels = torch.empty((0,), dtype=torch.int64, device=device)
    else:
        boxes = torch.as_tensor(detections.xyxy, dtype=torch.float32, device=device)
        conf = detections.confidence
        if conf is None:
            conf = np.ones(len(detections), dtype=np.float32)
        scores = torch.as_tensor(conf, dtype=torch.float32, device=device)
        cls = detections.class_id
        if cls is None:
            cls = np.zeros(len(detections), dtype=np.int64)
        labels = torch.as_tensor(cls, dtype=torch.int64, device=device)
    result = {"boxes": boxes, "scores": scores, "labels": labels}
    if getattr(detections, "tracker_id", None) is not None:
        tracker_ids = torch.as_tensor(
            detections.tracker_id, dtype=torch.int64, device=device
        )
        result["track_ids"] = tracker_ids
    return result

    
def update_track_metadata(tracked, metadata):
    if tracked is None or getattr(tracked, "tracker_id", None) is None:
        return
    tracker_ids = tracked.tracker_id
    if tracker_ids is None or len(tracker_ids) == 0:
        return
    confidences = tracked.confidence
    class_ids = tracked.class_id
    for idx, tracker_id in enumerate(tracker_ids):
        if tracker_id == -1:
            continue
        score = 1.0
        if confidences is not None and len(confidences) > idx:
            score = float(confidences[idx])
        cls_id = 0
        if class_ids is not None and len(class_ids) > idx:
            cls_id = int(class_ids[idx])
        metadata[int(tracker_id)] = {"class_id": cls_id, "score": score}


def tracker_predict_detections(tracker, metadata, default_class_id=0):
    tracked_tracks = getattr(tracker, "tracked_tracks", [])
    if not tracked_tracks:
        detections = sv.Detections.empty()
        detections.confidence = np.empty((0,), dtype=np.float32)
        detections.class_id = np.empty((0,), dtype=np.int64)
        detections.tracker_id = np.empty((0,), dtype=np.int64)
        return detections

    invalid_id = tracker.external_id_counter.NO_ID
    active_tracks = [
        track
        for track in tracked_tracks
        if track.is_activated and track.external_track_id != invalid_id
    ]

    if len(active_tracks) == 0:
        detections = sv.Detections.empty()
        detections.confidence = np.empty((0,), dtype=np.float32)
        detections.class_id = np.empty((0,), dtype=np.int64)
        detections.tracker_id = np.empty((0,), dtype=np.int64)
        return detections

    tracker.frame_id += 1
    STrack.multi_predict(active_tracks, tracker.shared_kalman)

    boxes = []
    confidences = []
    class_ids = []
    tracker_ids = []
    for track in active_tracks:
        tracker_id = int(track.external_track_id)
        boxes.append(track.tlbr)
        tracker_ids.append(tracker_id)
        meta = metadata.get(tracker_id)
        if meta is not None:
            class_ids.append(int(meta.get("class_id", default_class_id)))
            confidences.append(float(meta.get("score", 1.0)))
        else:
            class_ids.append(default_class_id)
            track_score = getattr(track, "score", None)
            confidences.append(float(track_score) if track_score is not None else 1.0)
        track.frame_id = tracker.frame_id

    detections = sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32),
        confidence=np.asarray(confidences, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=np.int64),
    )
    detections.tracker_id = np.asarray(tracker_ids, dtype=np.int64)
    return detections


def compute_iou_matrix(boxes_a, boxes_b):
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return torch.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=torch.float32)
    a = torch.as_tensor(boxes_a, dtype=torch.float32)
    b = torch.as_tensor(boxes_b, dtype=torch.float32)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def greedy_match(iou_matrix, threshold):
    matches = []
    if iou_matrix.numel() == 0:
        return matches
    matrix = iou_matrix.clone()
    while True:
        max_val = torch.max(matrix)
        if max_val < threshold:
            break
        flat_idx = torch.argmax(matrix)
        num_cols = matrix.shape[1]
        pred_idx = flat_idx // num_cols
        det_idx = flat_idx % num_cols
        matches.append((pred_idx.item(), det_idx.item(), max_val.item()))
        matrix[pred_idx, :] = -1
        matrix[:, det_idx] = -1
    return matches


def compute_speed_delta(prev_box, pred_box, det_box):
    def center(box):
        return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

    prev_center = center(prev_box)
    pred_center = center(pred_box)
    det_center = center(det_box)
    pred_speed = (pred_center[0] - prev_center[0], pred_center[1] - prev_center[1])
    det_speed = (det_center[0] - prev_center[0], det_center[1] - prev_center[1])
    diff = (pred_speed[0] - det_speed[0], pred_speed[1] - det_speed[1])
    return math.sqrt(diff[0] ** 2 + diff[1] ** 2)


def evaluate_safety_environment(predicted, detections, prev_track_boxes, safety_cfg):
    metrics = {}
    if predicted is None or len(predicted) == 0 or len(detections) == 0:
        metrics["reason"] = "insufficient_inputs"
        return False, metrics

    pred_boxes = predicted.xyxy
    det_boxes = detections.xyxy
    iou_matrix = compute_iou_matrix(pred_boxes, det_boxes)
    matching_iou_threshold = safety_cfg.get("matching_iou_threshold", 0.5)
    matches = greedy_match(iou_matrix, matching_iou_threshold)
    if len(matches) == 0:
        metrics["reason"] = "no_matches"
        return False, metrics

    avg_iou = float(np.mean([match[2] for match in matches]))
    detection_gap = (len(det_boxes) - len(matches)) / max(1, len(det_boxes))
    track_gap = (len(pred_boxes) - len(matches)) / max(1, len(pred_boxes))

    speed_diffs = []
    tracker_ids = getattr(predicted, "tracker_id", None)
    if tracker_ids is not None:
        for pred_idx, det_idx, _ in matches:
            if len(tracker_ids) <= pred_idx:
                continue
            tracker_id = int(tracker_ids[pred_idx])
            prev_box = prev_track_boxes.get(tracker_id)
            if prev_box is None:
                continue
            speed_diffs.append(
                compute_speed_delta(
                    prev_box,
                    pred_boxes[pred_idx],
                    det_boxes[det_idx],
                )
            )

    avg_speed_delta = float(np.mean(speed_diffs)) if speed_diffs else None

    metrics.update(
        {
            "avg_iou": avg_iou,
            "detection_gap": detection_gap,
            "track_gap": track_gap,
            "avg_speed_delta": avg_speed_delta,
        }
    )

    iou_ok = avg_iou >= safety_cfg.get("avg_iou_threshold", 0.8)
    det_gap_ok = detection_gap <= safety_cfg.get("max_detection_miss_rate", 0.1)
    track_gap_ok = track_gap <= safety_cfg.get("max_track_miss_rate", 0.1)
    speed_threshold = safety_cfg.get("speed_change_threshold", 5.0)
    speed_ok = (
        avg_speed_delta is not None and avg_speed_delta <= speed_threshold
    )

    is_safe = iou_ok and det_gap_ok and track_gap_ok and speed_ok
    metrics["is_safe"] = is_safe
    return is_safe, metrics


def val_pass(device, model, data, config, output_file):
    model.counting()
    model.clear_counts()
    model.eval()
    n_frames = 0
    outputs = []
    labels = []
    total_sparsity = 0 
    model_latency = 0
    system_latency = 0
    tracker_latency = 0
    memory = 0
    n_items = config.get("n_items", len(data))
    count = 0
    model_frame_count = 0
    MB = 1024 * 1024

    img_shape = config["model"]["input_shape"][-2:]    

    # Latency calculation
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # --------------- GPU Warmup for correct latency calculation ---------------
    vid_item = DataLoader(data[0], batch_size=1, collate_fn=collate_fn)
    model.reset()
    # model.backbone.reset_self()
    for i, (frame, annotations) in enumerate(vid_item):
        with torch.inference_mode():
            # images, x = model.pre_backbone(frame.to(device))
            results, _ = model(frame.to(device))
    model.clear_counts()
    # -------------------------------------------------------------------------
    short_edge_length = vid_item.dataset.combined_transform.short_edge_length
    max_size = vid_item.dataset.combined_transform.max_size
    mask_index_static_cpu = get_region_mask_static(
        short_edge_length=short_edge_length,
        max_size=max_size,
        region_sparsity=1 - config["sparsity"],
    )  # sparsity is keep rate
    tracker_cfg = config.get("tracker", {})
    tracker = sv.ByteTrack(
        track_activation_threshold=tracker_cfg.get(
            "track_activation_threshold", 0.5
        ),
        lost_track_buffer=tracker_cfg.get("lost_track_buffer", 30),
        minimum_matching_threshold=tracker_cfg.get(
            "minimum_matching_threshold", 0.8
        ),
        frame_rate=tracker_cfg.get("frame_rate", config.get("frame_rate", 30)),
        minimum_consecutive_frames=tracker_cfg.get(
            "minimum_consecutive_frames", 1
        ),
    )
    region_size = config.get("region_size", 16)
    total_region_tokens = (
        (img_shape[0] // region_size) * (img_shape[1] // region_size)
    )
    mask_margin = int(config.get("margin", 0))
    safety_cfg = config.get("safety", {})

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        step = 0
        n_frames += len(vid_item)
        model.reset()
        tracker.reset()
        track_metadata = {}
        prev_results_for_mask = None
        safe_tracker_only = False
        prev_detection_boxes = {}
        for frame, annotations in vid_item:
            with torch.inference_mode():
                is_key_frame = (step % config["period"]) == 0
                invalid_track_id = tracker.external_id_counter.NO_ID
                has_active_tracks = any(
                    track.is_activated and track.external_track_id != invalid_track_id
                    for track in getattr(tracker, "tracked_tracks", [])
                )
                if not has_active_tracks:
                    safe_tracker_only = False

                run_model = True
                mask_index_tensor = None
                sparsity = 0

                if is_key_frame:
                    safe_tracker_only = False
                else:
                    if safe_tracker_only and has_active_tracks:
                        run_model = False

                if run_model and not is_key_frame:
                    mask_index_cpu = get_region_mask_from_results(
                        prev_results_for_mask,
                        image_shape=tuple(img_shape),
                        conf_threshold=config.get("conf", 0.5),
                        region_size=region_size,
                        margin=mask_margin,
                    )
                    if mask_index_cpu is not None and config["sparsity"] < 1.0:
                        mask_index_cpu = (
                            torch.unique(
                                torch.cat((mask_index_cpu, mask_index_static_cpu), dim=1)
                            )
                            .reshape(1, -1)
                        )
                    elif mask_index_cpu is None and config["sparsity"] < 1.0:
                        mask_index_cpu = mask_index_static_cpu

                    if mask_index_cpu is not None:
                        keep_rate = mask_index_cpu.shape[1] / total_region_tokens
                        sparsity = 1 - keep_rate
                        mask_index_tensor = mask_index_cpu.to(device)

                mask_index = mask_index_tensor if run_model else None

                system_start = perf_counter()
                if run_model:
                    starter.record()
                    results, _ = model(frame.to(device), mask_index)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)

                    detections = results_to_supervision_detections(results[0])

                    safety_enabled = safety_cfg.get("enabled", True)
                    predicted_eval = None
                    if safety_enabled and has_active_tracks:
                        try:
                            tracker_snapshot = copy.deepcopy(tracker)
                            metadata_snapshot = copy.deepcopy(track_metadata)
                            predicted_eval = tracker_predict_detections(
                                tracker_snapshot, metadata_snapshot
                            )
                        except Exception:
                            predicted_eval = None

                    tracker_start = perf_counter()
                    tracked = tracker.update_with_detections(detections)
                    tracker_latency += (perf_counter() - tracker_start) * 1000
                    outputs.extend(results)
                    prev_results_for_mask = detach_results_for_mask(results)
                    update_track_metadata(tracked, track_metadata)
                    total_sparsity += sparsity
                    model_latency += curr_time
                    memory += torch.cuda.max_memory_allocated() / MB
                    model_frame_count += 1

                    prev_boxes_snapshot = {
                        track_id: box.copy() for track_id, box in prev_detection_boxes.items()
                    }

                    if safety_enabled and predicted_eval is not None:
                        is_safe, _ = evaluate_safety_environment(
                            predicted_eval,
                            detections,
                            prev_boxes_snapshot,
                            safety_cfg,
                        )
                        if is_safe and not safe_tracker_only:
                            safe_tracker_only = True
                            # print("Safe environment detected. Tracker only until next key frame.")

                    if getattr(tracked, "tracker_id", None) is not None:
                        for idx, tracker_id in enumerate(tracked.tracker_id):
                            if tracker_id == -1:
                                continue
                            prev_detection_boxes[int(tracker_id)] = tracked.xyxy[idx].copy()
                else:
                    tracker_start = perf_counter()
                    tracked = tracker_predict_detections(tracker, track_metadata)
                    tracker_latency += (perf_counter() - tracker_start) * 1000
                    outputs.append(detections_to_result_dict(tracked, device))
                system_latency += (perf_counter() - system_start) * 1000

                step += 1
                count += 1

            labels.extend([dict_to_device(annotation, device) for annotation in annotations])

    counts = model.total_counts() / n_frames

    model_frames = max(model_frame_count, 1)
    total_frames = max(count, 1)
    avg_model_sparsity = total_sparsity / model_frames
    tee_print(f'Sparsity: {avg_model_sparsity}', output_file)
    tee_print(f'Model latency (GPU): {model_latency / model_frames} ms', output_file)
    tee_print(f'System latency: {system_latency / total_frames} ms', output_file)
    tee_print(f'Tracker latency: {tracker_latency / total_frames} ms', output_file)
    tee_print(f'Memory: {memory / model_frames} MB', output_file)
    if config["model"]["backbone_config"]["backbone"] == "windowed":
        tee_print(f'GFLOPs: {sum([value for _, value in counts.items()]) / 1e9}', output_file)
    # MeanAveragePrecision is extremely slow. It seems fastest to call
    # update() and compute() just once, after all predictions are done.
    else:
        mask_index_test, _ = model.get_region_mask_static(region_sparsity=1 - avg_model_sparsity) # sparsity is keep rate
        _, x = model.pre_backbone(frame.to(device))
        macs = profile_macs(model.backbone, (x, None, 12, mask_index_test))
        tee_print(f'GFLOPs (torchprofile): {macs / 1e9}', output_file)

    mean_ap = MeanAveragePrecision().to(device)
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()
    recall_threshold = config.get("recall_iou_threshold", 0.5)
    recall_metrics = compute_detection_recall(outputs, labels, recall_threshold)

    model.clear_counts()
    return {"metrics": metrics, "recall": recall_metrics, "counts": counts}



def main():
    config = initialize_run(config_location=Path("./configs", "evaluate", "vitdet_vid"))
    long_edge = max(config["model"]["input_shape"][-2:])
    train_data = VID(
        Path("./data1", "vid_data"),
        split="vid_train",
        tar_path=Path("./data1", "vid_data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640 * long_edge // 1024, max_size=long_edge
        ),
    )
    val_data = VID(
        Path("./data1", "vid_data"),
        split="vid_val",
        tar_path=Path("./data1", "vid_data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640 * long_edge // 1024, max_size=long_edge
        ),
    )

    # run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)
    output_dir = Path(config["_output"])
    if config.get("evaluate", False):
        output_file = open(output_dir / "results.txt", 'a')
    else:
        output_file = open(output_dir / "output.txt", 'a')


    device = get_pytorch_device()
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    # Load and set up the model.
    config["model"]["mask"] = True
    model = ViTDet(**(config["model"]))
    msg = model.load_state_dict(torch.load(config["weights"]), strict=False)
    tee_print(msg, output_file)
    # tee_print(f"Keep rate:{model.backbone.keep_rate}", output_file)
    
    model = model.to(device)

    tee_print("Evaluating....", output_file)
    results = val_pass(device, model, val_data, config, output_file)
    
    # Print and save results.
    if isinstance(results, dict):
        for key, val in results.items():
            tee_print(key.capitalize(), output_file)
            tee_print(dict_string(val), output_file)
    else:
        tee_print(results, output_file)
    tee_print("", output_file) 

    output_file.close()

if __name__ == "__main__":
    main()
