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


def get_region_mask_dynamic(results, image_shape, conf_threshold=0.5, region_size=16, margin=0):
    if results is None:
        return None

    height, width = image_shape
    mask = torch.zeros((height, width), dtype=torch.float32)
    margin = int(max(margin, 0))
    has_bbox = False

    for result in results:
        boxes = result["boxes"].detach().cpu()
        scores = result["scores"].detach().cpu()
        for bbox, score in zip(boxes, scores):
            if score <= conf_threshold:
                continue
            has_bbox = True
            x1, y1, x2, y2 = bbox.tolist()
            x1 = max(0, int(np.floor(x1)) - margin)
            y1 = max(0, int(np.floor(y1)) - margin)
            x2 = min(width, int(np.ceil(x2)) + margin)
            y2 = min(height, int(np.ceil(y2)) + margin)
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1:y2, x1:x2] = 1

    if not has_bbox or torch.count_nonzero(mask) == 0:
        return None

    weight = torch.ones(1, 1, region_size, region_size)
    y = F.conv2d(mask.unsqueeze(0).unsqueeze(0), weight, stride=region_size)
    mask_index = torch.nonzero(y.flatten() > 0)
    if mask_index.numel() == 0:
        return None
    return mask_index.reshape(1, -1), y.numel()


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


def tracks_to_detections(tracks, track_metadata):
    boxes, scores, class_ids, tracker_ids = [], [], [], []
    for track in tracks:
        if not getattr(track, "is_activated", False):
            continue
        track_id = getattr(track, "external_track_id", -1)
        if track_id is None or track_id < 0:
            continue
        boxes.append(track.tlbr)
        metadata = track_metadata.get(int(track_id), {})
        scores.append(float(metadata.get("score", getattr(track, "score", 1.0))))
        class_ids.append(int(metadata.get("class_id", 0)))
        tracker_ids.append(int(track_id))

    if not boxes:
        detections = sv.Detections.empty()
        detections.confidence = np.empty((0,), dtype=np.float32)
        detections.class_id = np.empty((0,), dtype=np.int64)
        detections.tracker_id = np.empty((0,), dtype=np.int64)
        return detections

    detections = sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32),
        confidence=np.asarray(scores, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=np.int64),
    )
    detections.tracker_id = np.asarray(tracker_ids, dtype=np.int64)
    return detections


def update_tracker_metadata(track_metadata, detections):
    if detections is None or len(detections) == 0:
        return
    tracker_ids = getattr(detections, "tracker_id", None)
    if tracker_ids is None:
        return
    conf = detections.confidence
    cls = detections.class_id
    for idx, tracker_id in enumerate(tracker_ids):
        track_metadata[int(tracker_id)] = {
            "score": float(conf[idx]) if conf is not None else 1.0,
            "class_id": int(cls[idx]) if cls is not None else 0,
        }


def tracker_step_without_detections(tracker, track_metadata):
    empty = np.zeros((0, 5), dtype=np.float32)
    tracks = tracker.update_with_tensors(empty)
    return tracks_to_detections(tracks, track_metadata)


def simulate_tracker_prediction(tracker, track_metadata):
    tracker_clone = copy.deepcopy(tracker)
    return tracker_step_without_detections(tracker_clone, track_metadata)


def compute_iou_matrix(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter_area
    union = np.clip(union, a_min=1e-6, a_max=None)
    return inter_area / union


def greedy_match(iou_matrix, threshold):
    if iou_matrix.size == 0:
        return []
    matrix = iou_matrix.copy()
    matches = []
    while True:
        max_index = np.unravel_index(np.argmax(matrix), matrix.shape)
        max_value = matrix[max_index]
        if max_value < threshold:
            break
        matches.append((max_index[0], max_index[1], float(max_value)))
        matrix[max_index[0], :] = -1
        matrix[:, max_index[1]] = -1
    return matches


def evaluate_safe_environment(predicted, actual, thresholds):
    cfg = {
        "iou_threshold": thresholds.get("iou_threshold", 0.8),
        "speed_threshold": thresholds.get("speed_threshold", 0.05),
        "unmatched_threshold": thresholds.get("unmatched_threshold", 0.1),
        "match_iou_threshold": thresholds.get(
            "match_iou_threshold", thresholds.get("iou_threshold", 0.8)
        ),
    }

    num_pred = len(predicted)
    num_actual = len(actual)
    metrics = {
        "avg_iou": 0.0,
        "avg_speed_change": 0.0,
        "new_detection_rate": 1.0 if num_actual > 0 else 0.0,
        "missing_track_rate": 1.0 if num_pred > 0 else 0.0,
        "num_pred": num_pred,
        "num_actual": num_actual,
        "matched_pairs": 0,
        "is_safe": False,
    }

    if num_pred == 0 and num_actual == 0:
        metrics.update(
            {
                "avg_iou": 1.0,
                "avg_speed_change": 0.0,
                "new_detection_rate": 0.0,
                "missing_track_rate": 0.0,
                "matched_pairs": 0,
                "is_safe": True,
            }
        )
        return metrics

    if num_pred == 0 or num_actual == 0:
        return metrics

    iou_matrix = compute_iou_matrix(predicted.xyxy, actual.xyxy)
    matches = greedy_match(iou_matrix, cfg["match_iou_threshold"])
    metrics["matched_pairs"] = len(matches)

    if not matches:
        return metrics

    matched_ious = [match[2] for match in matches]
    metrics["avg_iou"] = float(np.mean(matched_ious))

    speed_changes = []
    for pred_idx, actual_idx, _ in matches:
        pred_box = predicted.xyxy[pred_idx]
        actual_box = actual.xyxy[actual_idx]
        pred_center = np.array(
            [(pred_box[0] + pred_box[2]) * 0.5, (pred_box[1] + pred_box[3]) * 0.5],
            dtype=np.float32,
        )
        actual_center = np.array(
            [
                (actual_box[0] + actual_box[2]) * 0.5,
                (actual_box[1] + actual_box[3]) * 0.5,
            ],
            dtype=np.float32,
        )
        diag = np.sqrt(
            max(
                (actual_box[2] - actual_box[0]) ** 2
                + (actual_box[3] - actual_box[1]) ** 2,
                1e-6,
            )
        )
        center_delta = np.linalg.norm(pred_center - actual_center)
        speed_changes.append(center_delta / diag)

    metrics["avg_speed_change"] = float(np.mean(speed_changes)) if speed_changes else 0.0
    metrics["new_detection_rate"] = (
        (num_actual - len(matches)) / max(1, num_actual)
    )
    metrics["missing_track_rate"] = (num_pred - len(matches)) / max(1, num_pred)

    metrics["is_safe"] = (
        metrics["avg_iou"] >= cfg["iou_threshold"]
        and metrics["avg_speed_change"] <= cfg["speed_threshold"]
        and metrics["new_detection_rate"] <= cfg["unmatched_threshold"]
        and metrics["missing_track_rate"] <= cfg["unmatched_threshold"]
    )
    return metrics
    
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
    mask_index_static = get_region_mask_static(
        short_edge_length=short_edge_length,
        max_size=max_size,
        region_sparsity=1 - config["sparsity"],
    )

    tracker_cfg = config.get("tracker", {})
    tracker_input_threshold = tracker_cfg.get(
        "input_score_threshold",
        tracker_cfg.get("track_activation_threshold", 0.5),
    )
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
    margin = int(config.get("margin", 0))
    conf_threshold = float(config.get("conf", 0.5))

    safe_cfg = config.get("safe_env", {})
    safe_enabled = safe_cfg.get("enabled", True)
    safe_log = safe_cfg.get("log", True)

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        step = 0
        n_frames += len(vid_item)
        model.reset()
        tracker.reset()
        track_metadata = {}
        last_model_results = None
        safe_skip_active = False

        for frame, annotations in vid_item:
            with torch.inference_mode():
                is_key_frame = (step % config["period"]) == 0
                if is_key_frame:
                    safe_skip_active = False
                run_model = is_key_frame or not safe_skip_active

                mask_index = None
                sparsity = 0
                if run_model:
                    mask_index_cpu = None
                    dynamic_mask = None
                    if not is_key_frame and last_model_results is not None:
                        dynamic_mask = get_region_mask_dynamic(
                            last_model_results,
                            image_shape=tuple(img_shape),
                            conf_threshold=conf_threshold,
                            region_size=region_size,
                            margin=margin,
                        )
                    if dynamic_mask is not None:
                        mask_index_cpu, _ = dynamic_mask
                        if config["sparsity"] < 1.0:
                            mask_index_cpu = torch.unique(
                                torch.cat((mask_index_cpu, mask_index_static), dim=1)
                            ).reshape(1, -1)
                    elif config["sparsity"] < 1.0:
                        mask_index_cpu = mask_index_static

                    if mask_index_cpu is not None:
                        keep_rate = mask_index_cpu.shape[1] / max(1, total_region_tokens)
                        mask_index = mask_index_cpu.to(device)
                        sparsity = 1 - keep_rate

                system_start = perf_counter()
                if run_model:
                    starter.record()
                    results, _ = model(frame.to(device), mask_index)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    model_latency += curr_time
                    last_model_results = results

                    detections_full = results_to_supervision_detections(results[0])
                    detections_for_tracker = detections_full
                    if (
                        tracker_input_threshold is not None
                        and len(detections_for_tracker) > 0
                    ):
                        keep = (
                            detections_for_tracker.confidence
                            > tracker_input_threshold
                        )
                        detections_for_tracker = detections_for_tracker[keep]

                    predicted_safe = None
                    if is_key_frame and safe_enabled:
                        predicted_safe = simulate_tracker_prediction(
                            tracker, track_metadata
                        )

                    tracker_start = perf_counter()
                    tracked = tracker.update_with_detections(detections_for_tracker)
                    tracker_latency += (perf_counter() - tracker_start) * 1000
                    system_latency += (perf_counter() - system_start) * 1000
                    update_tracker_metadata(track_metadata, tracked)
                    outputs.extend(results)

                    if is_key_frame and safe_enabled:
                        safe_metrics = evaluate_safe_environment(
                            predicted_safe, detections_full, safe_cfg
                        )
                        safe_skip_active = safe_metrics["is_safe"]
                        if safe_log:
                            status = "ON" if safe_skip_active else "OFF"
                            tee_print(
                                (
                                    f"[SAFE {status}] IoU={safe_metrics['avg_iou']:.3f} "
                                    f"Speed={safe_metrics['avg_speed_change']:.3f} "
                                    f"NewDet={safe_metrics['new_detection_rate']:.3f} "
                                    f"MissTrack={safe_metrics['missing_track_rate']:.3f}"
                                ),
                                output_file,
                            )

                    step += 1
                    count += 1
                    total_sparsity += sparsity
                    MB = 1024 * 1024
                    memory += torch.cuda.max_memory_allocated() / MB
                else:
                    tracker_start = perf_counter()
                    predicted_detections = tracker_step_without_detections(
                        tracker, track_metadata
                    )
                    tracker_latency += (perf_counter() - tracker_start) * 1000
                    system_latency += (perf_counter() - system_start) * 1000
                    outputs.append(
                        detections_to_result_dict(predicted_detections, device)
                    )
                    step += 1
                    count += 1
                    total_sparsity += sparsity

            labels.extend(
                [dict_to_device(annotation, device) for annotation in annotations]
            )

    counts = model.total_counts() / n_frames

    tee_print(f'Sparsity: {total_sparsity / count}', output_file)
    tee_print(f'Model latency (GPU): {model_latency / count} ms', output_file)
    tee_print(f'System latency: {system_latency / count} ms', output_file)
    tee_print(f'Tracker latency: {tracker_latency / count} ms', output_file)
    tee_print(f'Memory: {memory / count} MB', output_file)
    if config["model"]["backbone_config"]["backbone"] == "windowed":
        tee_print(f'GFLOPs: {sum([value for _, value in counts.items()]) / 1e9}', output_file)
    # MeanAveragePrecision is extremely slow. It seems fastest to call
    # update() and compute() just once, after all predictions are done.
    else:
        mask_index_test, _ = model.get_region_mask_static(region_sparsity=1 - total_sparsity/count) # sparsity is keep rate
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
