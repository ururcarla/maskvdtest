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
from utils.misc import dict_to_device, squeeze_dict, tee_print, compute_detection_recall


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])
    return tuple(batch)


def evaluate_vitdet_metrics(device, model, data, config, output_file):
    model.counting()
    model.clear_counts()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))
    latency = 0
    memory = 0
    count = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    vid_item = DataLoader(data[0], batch_size=1, collate_fn=collate_fn)
    model.reset()
    for i, (frame, _) in enumerate(vid_item):
        with torch.inference_mode():
            _ = model(frame.to(device))

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        n_frames += len(vid_item)
        model.reset()
        for frame, annotations in vid_item:
            with torch.inference_mode():
                starter.record()
                results, _ = model(frame.to(device))
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)

                outputs.extend(results)
                count += 1
                latency += curr_time
                MB = 1024 * 1024
                memory += torch.cuda.max_memory_allocated() / MB

            labels.extend([dict_to_device(annotation, device) for annotation in annotations])

    tee_print(f"Latency: {latency / count} ms", output_file)
    tee_print(f"Memory: {memory / count} MB", output_file)

    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()
    recall_threshold = config.get("recall_iou_threshold", 0.5)
    recall_metrics = compute_detection_recall(outputs, labels, recall_threshold)

    counts = model.total_counts() / n_frames
    model.clear_counts()
    return {"metrics": metrics, "recall": recall_metrics, "counts": counts}


def main():
    config = initialize_run(config_location=Path("configs", "evaluate", "argoverse_hd"))
    long_edge = max(config["model"]["input_shape"][-2:])
    dataset_cfg = config.get("dataset", {})
    dataset_root = Path(dataset_cfg.get("root", "./data/argoversehd"))
    split = dataset_cfg.get("split", config.get("split", "val"))
    ann_path = dataset_cfg.get("annotations")

    data = ArgoverseHD(
        location=dataset_root,
        split=split,
        annotations_path=Path(ann_path) if ann_path is not None else None,
        combined_transform=VIDResize(
            short_edge_length=640 * long_edge // 1024,
            max_size=long_edge,
        ),
    )

    run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)


if __name__ == "__main__":
    main()

