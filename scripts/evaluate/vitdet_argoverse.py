#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from datasets.builders import build_video_dataset
from models.vitdet import ViTDet
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import compute_detection_recall, dict_to_device, tee_print


def collate_fn(batch):
    frames, annotations = zip(*batch)
    frames = torch.stack(frames)
    return frames, annotations


def build_dataset(config):
    dataset_cfg = config.get("dataset") or {}
    if not dataset_cfg:
        raise ValueError("配置中缺少 dataset 节点，无法构建 Argoverse 数据集。")
    split = dataset_cfg.get("split") or dataset_cfg.get("val_split") or "val"
    long_edge = max(config["model"]["input_shape"][-2:])
    return build_video_dataset(config, split=split, long_edge=long_edge, shuffle=False)


def evaluate_vitdet_metrics(device, model, data, config, output_file):
    model.counting()
    model.clear_counts()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))
    latency = 0.0
    memory = 0.0
    count = 0

    use_cuda_events = device == "cuda" and torch.cuda.is_available()
    starter = ender = None
    if use_cuda_events:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        warm_loader = DataLoader(data[0], batch_size=1, collate_fn=collate_fn)
        model.reset()
        for frame, _ in warm_loader:
            with torch.inference_mode():
                model(frame.to(device))
        torch.cuda.synchronize()

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_loader = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        n_frames += len(vid_loader)
        model.reset()
        if config.get("mask") == "static":
            mask_index, _ = model.get_region_mask_static(region_sparsity=1 - config["sparsity"])
        else:
            mask_index = None

        for frame, annotations in vid_loader:
            with torch.inference_mode():
                if use_cuda_events:
                    starter.record()
                results, _ = model(frame.to(device), mask_index)
                if use_cuda_events:
                    ender.record()
                    torch.cuda.synchronize()
                    latency += starter.elapsed_time(ender)
                    memory += torch.cuda.max_memory_allocated() / (1024 * 1024)
                outputs.extend(results)
                count += 1
            labels.extend([dict_to_device(annotation, device) for annotation in annotations])

    if use_cuda_events and count > 0:
        tee_print(f"Latency: {latency / count:.3f} ms", output_file)
        tee_print(f"Memory: {memory / count:.2f} MB", output_file)

    mean_ap = MeanAveragePrecision()
    if outputs:
        mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()
    recall_threshold = config.get("recall_iou_threshold", 0.5)
    recall_metrics = compute_detection_recall(outputs, labels, recall_threshold)

    counts = model.total_counts() / max(n_frames, 1)
    model.clear_counts()
    return {"metrics": metrics, "recall": recall_metrics, "counts": counts}


def main():
    config = initialize_run(config_location=Path("configs", "evaluate", "vitdet_vid"))
    data = build_dataset(config)
    run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)


if __name__ == "__main__":
    main()

