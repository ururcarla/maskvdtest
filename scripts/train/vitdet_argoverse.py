#!/usr/bin/env python3

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from detectron2.data.detection_utils import annotations_to_instances, BoxMode
from detectron2.utils.events import EventStorage
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from backbones.base import dict_string
from backbones.policies import TokenNormTopK
from datasets.builders import build_video_dataset
from models.vitdet import ViTDet
from util.lr_sched import LR_Scheduler
from utils.config import initialize_run
from utils.misc import dict_to_device, get_pytorch_device, set_policies, tee_print


def collate_fn(batch):
    frames, annotations = zip(*batch)
    frames = torch.stack(frames)
    return frames, annotations


def _estimate_total_frames(dataset, n_items):
    video_info = getattr(dataset, "video_info", None)
    if not video_info:
        return max(len(dataset), n_items)
    total = 0
    limit = min(n_items, len(video_info))
    for idx in range(limit):
        frames = video_info[idx].get("frames")
        total += len(frames) if isinstance(frames, list) else 0
    return max(total, n_items)


def build_datasets(config):
    dataset_cfg = config.get("dataset") or {}
    if not dataset_cfg:
        raise ValueError("配置中缺少 dataset 节点，无法构建 Argoverse 数据集。")
    train_split = dataset_cfg.get("train_split") or dataset_cfg.get("split") or "train"
    val_split = dataset_cfg.get("val_split") or dataset_cfg.get("split") or "val"
    long_edge = max(config["model"]["input_shape"][-2:])
    train_data = build_video_dataset(config, split=train_split, long_edge=long_edge, shuffle=True)
    val_data = build_video_dataset(config, split=val_split, long_edge=long_edge, shuffle=False)
    return train_data, val_data


def train_pass(config, device, epoch, model, optimizer, lr_sched, data, tensorboard, output_file):
    model.train()
    n_items = config.get("n_items", len(data))
    accum_iter = config["accum_iter"]
    total_loss = 0.0
    frames_processed = 0
    total_frames = _estimate_total_frames(data, n_items)

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_loader = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        for frame, annotations in vid_loader:
            frames_processed += 1
            annotation_list = []
            gt_instances = []
            for annotation in annotations:
                for i, bbox in enumerate(annotation["boxes"]):
                    annotation_list.append(
                        {
                            "bbox": bbox,
                            "category_id": annotation["labels"][i],
                            "bbox_mode": BoxMode.XYXY_ABS,
                        }
                    )
                gt_instance = annotations_to_instances(
                    annotation_list, frame.shape[-2:], frame.shape[-2:]
                )
                gt_instances.append(gt_instance.to(device))

            if frames_processed % accum_iter == 0:
                progress = min(1.0, frames_processed / max(total_frames, 1))
                epoch_position = (epoch - 1) + progress
                lr_sched.adjust_learning_rate(epoch_position)

            with EventStorage():
                images, x = model.pre_backbone(frame.to(device))
                if config["mask"] == "static":
                    mask_index, _ = model.get_region_mask_static(
                        region_sparsity=1 - config["sparsity"]
                    )
                    x = model.backbone(x, mask_id=mask_index)
                else:
                    x = model.backbone(x)
                x = x.transpose(-1, -2)
                x = x.view(x.shape[:-1] + model.backbone_input_size)
                x = model.pyramid(x)
                x = dict(zip(model.proposal_generator.in_features, x))
                proposals, proposal_losses = model.proposal_generator(images, x, gt_instances)
                _, detector_losses = model.roi_heads(images, x, proposals, gt_instances)

            losses = {**detector_losses, **proposal_losses}
            loss = sum(losses.values())
            # Argoverse 训练会在 Video item 内多次复用模块缓存，需保留计算图
            loss.backward(retain_graph=True)
            total_loss += loss.item()

            if frames_processed % accum_iter == 0:
                tee_print(
                    f"Loss: {total_loss / frames_processed:.6f}, lr: {optimizer.param_groups[0]['lr']:.6e}",
                    output_file,
                )
                optimizer.step()
                optimizer.zero_grad()

            if tensorboard is not None:
                tensorboard.add_scalar("train/loss", loss.item(), global_step=frames_processed)


def val_pass(device, model, data, config):
    model.counting()
    model.clear_counts()
    model.eval()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_loader = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        n_frames += len(vid_loader)
        model.reset()
        if config["mask"] == "static":
            mask_index, _ = model.get_region_mask_static(region_sparsity=1 - config["sparsity"])
        else:
            mask_index = None

        for frame, annotations in vid_loader:
            with torch.inference_mode():
                results, _ = model(frame.to(device), mask_index)
            pred_batch = list(results)
            label_batch = [dict_to_device(annotation, device) for annotation in annotations]
            if len(pred_batch) != len(label_batch):
                min_len = min(len(pred_batch), len(label_batch))
                print(
                    f"[val_pass] Warning: preds ({len(pred_batch)}) and targets ({len(label_batch)}) length mismatch. "
                    f"Truncating to {min_len}."
                )
                pred_batch = pred_batch[:min_len]
                label_batch = label_batch[:min_len]
            outputs.extend(pred_batch)
            labels.extend(label_batch)

    mean_ap = MeanAveragePrecision()
    if outputs:
        mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()
    counts = model.total_counts() / max(n_frames, 1)
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def train_vitdet(config, model, device, train_data, val_data, output_file):
    optimizer_class = getattr(optim, config["optimizer"])
    optimizer = optimizer_class(model.parameters(), **config["optimizer_kwargs"])
    lr_sched = LR_Scheduler(
        optimizer,
        config["lr_scheduler_kwargs"]["warmup_epochs"],
        config["lr_scheduler_kwargs"]["min_lr"],
        config["optimizer_kwargs"]["lr"],
        config["epochs"],
    )

    if "tensorboard" in config:
        base_name = config["tensorboard"]
        now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tensorboard = SummaryWriter(f"{base_name}_{now_str}")
    else:
        tensorboard = None

    n_epochs = config["epochs"]
    for epoch in range(n_epochs):
        tee_print(f"\nEpoch {epoch + 1}/{n_epochs}", file=output_file)
        train_pass(config, device, epoch + 1, model, optimizer, lr_sched, train_data, tensorboard, output_file)
        results = val_pass(device, model, val_data, config)
        model.reset()

        if isinstance(results, dict):
            for key, val in results.items():
                tee_print(key.capitalize(), output_file)
                tee_print(dict_string(val), output_file)
        else:
            tee_print(results, output_file)
        tee_print("", output_file)

        weight_path = Path(config["_output"]) / f"weights_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), weight_path)
        tee_print(f"Saved weights to {weight_path}", output_file)

    if tensorboard is not None:
        tensorboard.close()

    final_path = Path(config["_output"]) / "weights_final.pth"
    torch.save(model.state_dict(), final_path)
    tee_print(f"Saved weights to {final_path}", output_file)


def main():
    config = initialize_run(config_location=Path("configs", "train", "vitdet_vid"))
    train_data, val_data = build_datasets(config)

    output_dir = Path(config["_output"])
    output_file = open(output_dir / "output.txt", "a")

    device = get_pytorch_device()
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    model = ViTDet(**config["model"])
    set_policies(model, TokenNormTopK, k=512)
    ckpt = torch.load(config["weights"], map_location="cpu")
    for key in list(ckpt.keys()):
        if key.startswith("roi_heads.box_predictor.cls_score") or key.startswith(
            "roi_heads.box_predictor.bbox_pred"
        ):
            del ckpt[key]
    msg = model.load_state_dict(ckpt, strict=False)
    tee_print(msg, output_file)
    model = model.to(device)

    if config.get("evaluate", False):
        tee_print("Evaluating....", output_file)
        results = val_pass(device, model, val_data, config)
        if isinstance(results, dict):
            for key, val in results.items():
                tee_print(key.capitalize(), output_file)
                tee_print(dict_string(val), output_file)
        else:
            tee_print(results, output_file)
        tee_print("", output_file)
    else:
        train_vitdet(config, model, device, train_data, val_data, output_file)

    output_file.close()


if __name__ == "__main__":
    main()