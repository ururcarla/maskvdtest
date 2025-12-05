#!/usr/bin/env python3

import sys
from pathlib import Path
from collections import deque

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
    mask = torch.zeros(image_shape)
    has_box = False
    for result in results:
        for i, bbox in enumerate(result['boxes']):
            if result['scores'][i] > conf_threshold:
                has_box = True
                x1, y1, x2, y2 = bbox
                x1 = max(0, int(x1) - margin)
                y1 = max(0, int(y1) - margin)
                x2 = min(image_shape[1], int(x2) + margin)
                y2 = min(image_shape[0], int(y2) + margin)
                if x2 <= x1 or y2 <= y1:
                    continue
                mask[y1:y2, x1:x2] = 1
    if not has_box:
        mask = torch.ones(image_shape)
    weight = torch.ones(1, 1, region_size, region_size)
    y = F.conv2d(mask.unsqueeze(0).unsqueeze(0), weight, stride=region_size)
    mask_index = torch.nonzero(y.flatten() > 0).reshape(1, -1)
    sparsity = 1 - mask_index.shape[1] / y.numel()
    return mask_index, sparsity, y.numel()


class SlidingWindowHeatmap:
    """Maintains a per-region heatmap with a sliding window of recent frames."""

    def __init__(self, image_shape, region_size, window_size, score_threshold=0.75):
        self.image_shape = image_shape
        self.region_size = max(1, int(region_size))
        self.grid_height = max(1, image_shape[0] // self.region_size)
        self.grid_width = max(1, image_shape[1] // self.region_size)
        self.window_size = max(1, int(window_size))
        self.score_threshold = 0.0 if score_threshold is None else max(0.0, float(score_threshold))
        self.accumulator = torch.zeros((1, 1, self.grid_height, self.grid_width), dtype=torch.float32)
        self.history = deque()
        self.total_activity = 0.0

    def _empty_map(self):
        return torch.zeros_like(self.accumulator)

    def _boxes_to_grid(self, boxes, scores=None):
        grid = self._empty_map()
        if boxes is None or boxes.numel() == 0:
            return grid
        boxes_cpu = boxes.detach().cpu().numpy()
        use_scores = (
            scores is not None
            and torch.is_tensor(scores)
            and scores.numel() == boxes.shape[0]
        )
        scores_cpu = scores.detach().cpu().numpy() if use_scores else None
        height, width = self.image_shape
        for idx, (x1, y1, x2, y2) in enumerate(boxes_cpu):
            if use_scores and scores_cpu[idx] <= self.score_threshold:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            x1 = float(max(0.0, min(x1, width)))
            x2 = float(max(0.0, min(x2, width)))
            y1 = float(max(0.0, min(y1, height)))
            y2 = float(max(0.0, min(y2, height)))
            start_col = int(np.floor(x1 / self.region_size))
            end_col = int(np.ceil(x2 / self.region_size))
            start_row = int(np.floor(y1 / self.region_size))
            end_row = int(np.ceil(y2 / self.region_size))
            start_col = max(0, min(start_col, self.grid_width))
            end_col = max(0, min(end_col, self.grid_width))
            start_row = max(0, min(start_row, self.grid_height))
            end_row = max(0, min(end_row, self.grid_height))
            if end_col <= start_col or end_row <= start_row:
                continue
            grid[:, :, start_row:end_row, start_col:end_col] += 1.0
        return grid

    def _push_map(self, region_map):
        self.history.append(region_map)
        self.accumulator += region_map
        if len(self.history) > self.window_size:
            removed = self.history.popleft()
            self.accumulator -= removed
        self.total_activity = float(self.accumulator.sum().item())

    def update(self, boxes, scores=None):
        if boxes is None or boxes.numel() == 0:
            region_map = self._empty_map()
        else:
            region_map = self._boxes_to_grid(boxes, scores)
        self._push_map(region_map)

    def build_mask_index(self, keep_ratio, min_ready_frames=1, min_activity=0.0):
        if keep_ratio <= 0.0:
            return None
        keep_ratio = min(1.0, float(keep_ratio))
        min_ready_frames = min(self.window_size, max(1, int(min_ready_frames)))
        if len(self.history) < min_ready_frames:
            return None
        if self.total_activity <= float(min_activity):
            return None
        return topkmask(self.accumulator, sparsity=keep_ratio)


def merge_mask_indices(*indices):
    masks = [idx for idx in indices if idx is not None]
    if not masks:
        return None
    merged = torch.cat(masks, dim=1)
    unique = torch.unique(merged.view(-1))
    if unique.numel() == 0:
        return None
    return unique.unsqueeze(0)


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
    mask_index_static = get_region_mask_static(short_edge_length=short_edge_length,
                                               max_size=max_size,
                                               region_sparsity=1 - config["sparsity"]) # sparsity is keep rate 

    region_size = config.get("region_size", 16)
    total_region_tokens = (
        (img_shape[0] // region_size) * (img_shape[1] // region_size)
    )
    dynamic_window = int(config.get("dynamic_heatmap_window", 30))
    dynamic_min_frames = int(config.get("dynamic_heatmap_min_frames", max(1, min(dynamic_window, 30))))
    dynamic_min_activity = float(config.get("dynamic_heatmap_min_activity", 0.0))
    dynamic_score_threshold = float(config.get("dynamic_heatmap_score_threshold", 0.75))
    dynamic_conf_threshold = float(config.get("conf", 0.5))
    dynamic_margin = int(config.get("margin", 0))

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        step = 0
        n_frames += len(vid_item)
        model.reset()
        heatmap_state = SlidingWindowHeatmap(
            image_shape=tuple(img_shape),
            region_size=region_size,
            window_size=dynamic_window,
            score_threshold=dynamic_score_threshold,
        )
        heatmap_mask_cache = None
        prev_results = None
        for frame, annotations in vid_item:
            with torch.inference_mode():
                is_key_frame = (step % config["period"]) == 0
                dynamic_mask = None
                if not is_key_frame and prev_results is not None:
                    dynamic_mask, _, _ = get_region_mask_dynamic(
                        prev_results,
                        image_shape=img_shape,
                        conf_threshold=dynamic_conf_threshold,
                        region_size=region_size,
                        margin=dynamic_margin,
                    )

                if is_key_frame:
                    mask_index = None
                    sparsity = 0
                else:
                    mask_candidates = []
                    if dynamic_mask is not None:
                        mask_candidates.append(dynamic_mask)
                    if heatmap_mask_cache is not None:
                        mask_candidates.append(heatmap_mask_cache)
                    if config["sparsity"] < 1.0:
                        mask_candidates.append(mask_index_static)

                    combined_mask = (
                        merge_mask_indices(*mask_candidates) if mask_candidates else None
                    )
                    if combined_mask is not None:
                        keep_rate = combined_mask.shape[1] / total_region_tokens
                        mask_index = combined_mask.to(device)
                        sparsity = 1 - keep_rate
                    else:
                        mask_index = None
                        sparsity = 0

                system_start = perf_counter()
                starter.record()
                results, _ = model(frame.to(device), mask_index)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)

                system_latency += (perf_counter() - system_start) * 1000
                outputs.extend(results)
                prev_results = results
                frame_results = results[0]
                heatmap_state.update(
                    frame_results["boxes"],
                    frame_results.get("scores"),
                )
                heatmap_mask = heatmap_state.build_mask_index(
                    keep_ratio=1-config["sparsity"],
                    min_ready_frames=dynamic_min_frames,
                    min_activity=dynamic_min_activity,
                )
                heatmap_mask_cache = heatmap_mask.cpu() if heatmap_mask is not None else None

                step += 1
                count += 1
                total_sparsity += sparsity
                model_latency += curr_time
                MB = 1024 * 1024
                memory += torch.cuda.max_memory_allocated() / MB

            # labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))
            labels.extend([dict_to_device(annotation, device) for annotation in annotations])

    counts = model.total_counts() / n_frames

    tee_print(f'Sparsity: {total_sparsity / count}', output_file)
    tee_print(f'Model latency (GPU): {model_latency / count} ms', output_file)
    tee_print(f'System latency: {system_latency / count} ms', output_file)
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
