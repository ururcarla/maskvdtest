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

from datasets.argoverse_hd import ArgoverseHD
from datasets.vid import VIDResize
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


def get_region_mask_from_tracks(tracked, image_shape, region_size=16, margin=0):
    if tracked is None or len(tracked) == 0 or getattr(tracked, "xyxy", None) is None:
        return None

    height, width = image_shape
    mask = torch.zeros((height, width), dtype=torch.float32)
    boxes = tracked.xyxy
    margin = int(max(margin, 0))

    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(np.floor(x1)) - margin)
        y1 = max(0, int(np.floor(y1)) - margin)
        x2 = min(width, int(np.ceil(x2)) + margin)
        y2 = min(height, int(np.ceil(y2)) + margin)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 1

    if torch.count_nonzero(mask) == 0:
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
    mask_index_static = get_region_mask_static(short_edge_length=short_edge_length,
                                               max_size=max_size,
                                               region_sparsity=1 - config["sparsity"]) # sparsity is keep rate 

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
    mask_index_static_device = mask_index_static.to(device)
    mask_margin = int(config.get("tracker_mask_margin", config.get("margin", 0)))

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        step = 0
        n_frames += len(vid_item)
        model.reset()
        tracker.reset()
        dynamic_mask_cache = None
        track_metadata = {}
        use_tracker_only_next = False
        for frame, annotations in vid_item:
            with torch.inference_mode():
                is_key_frame = (step % config["period"]) == 0
                invalid_track_id = tracker.external_id_counter.NO_ID
                has_active_tracks = any(
                    track.is_activated and track.external_track_id != invalid_track_id
                    for track in getattr(tracker, "tracked_tracks", [])
                )
                if is_key_frame:
                    mask_index = None
                    sparsity = 0
                    run_model = True
                    use_tracker_only_next = False
                else:
                    run_model = not use_tracker_only_next
                    use_tracker_only_next = not use_tracker_only_next
                    if not has_active_tracks:
                        run_model = True
                    if run_model:
                        mask_index = dynamic_mask_cache
                        if mask_index is not None:
                            keep_rate = mask_index.shape[1] / total_region_tokens
                            mask_index = mask_index.to(device)
                            sparsity = 1 - keep_rate
                        elif config["sparsity"] < 1.0:
                            mask_index = mask_index_static_device
                            keep_rate = mask_index.shape[1] / total_region_tokens
                            sparsity = 1 - keep_rate
                        else:
                            mask_index = None
                            sparsity = 0
                    else:
                        mask_index = None
                        sparsity = 0
               
                system_start = perf_counter()
                if run_model:
                    starter.record()
                    results, _ = model(frame.to(device), mask_index)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)

                    detections = results_to_supervision_detections(results[0])
                    tracker_start = perf_counter()
                    tracked = tracker.update_with_detections(detections)
                    tracker_latency += (perf_counter() - tracker_start) * 1000
                    outputs.extend(results)
                    update_track_metadata(tracked, track_metadata)
                    total_sparsity += sparsity
                    model_latency += curr_time
                    memory += torch.cuda.max_memory_allocated() / MB
                    model_frame_count += 1
                else:
                    tracker_start = perf_counter()
                    tracked = tracker_predict_detections(
                        tracker, track_metadata
                    )
                    tracker_latency += (perf_counter() - tracker_start) * 1000
                    outputs.append(detections_to_result_dict(tracked, device))
                system_latency += (perf_counter() - system_start) * 1000

                dynamic_mask_cache = get_region_mask_from_tracks(
                    tracked,
                    image_shape=tuple(img_shape),
                    region_size=region_size,
                    margin=mask_margin,
                )
                if dynamic_mask_cache is not None:
                    dynamic_mask_cache = dynamic_mask_cache.cpu()
                step += 1
                count += 1

            # labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))
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
    config = initialize_run(config_location=Path("./configs", "evaluate", "argoverse_hd"))
    long_edge = max(config["model"]["input_shape"][-2:])
    dataset_cfg = config.get("dataset", {})
    dataset_root = Path(dataset_cfg.get("root", "./data/argoversehd"))
    train_split = dataset_cfg.get("train_split", "train")
    val_split = dataset_cfg.get("split", config.get("split", "val"))
    train_ann = dataset_cfg.get("train_annotations")
    val_ann = dataset_cfg.get("annotations")

    def build_dataset(split_name, ann_path):
        return ArgoverseHD(
            location=dataset_root,
            split=split_name,
            annotations_path=Path(ann_path) if ann_path is not None else None,
            combined_transform=VIDResize(
                short_edge_length=640 * long_edge // 1024, max_size=long_edge
            ),
        )

    train_data = build_dataset(train_split, train_ann)
    val_data = build_dataset(val_split, val_ann)

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
