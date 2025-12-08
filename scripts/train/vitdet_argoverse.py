#!/usr/bin/env python3

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.cuda.amp import GradScaler, autocast
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


def _set_modules_trainable(model, module_names, trainable):
    for module_name in module_names:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = trainable


def _set_backbone_blocks_trainable(model, block_indices, trainable):
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        _set_modules_trainable(model, ["backbone"], trainable)
        return
    n_blocks = len(blocks)
    if n_blocks == 0:
        return
    unique_indices = []
    for idx in block_indices or []:
        if isinstance(idx, int):
            unique_indices.append(idx)
    index_set = set()
    for idx in unique_indices:
        if idx < 0:
            idx = n_blocks + idx
        if 0 <= idx < n_blocks and idx not in index_set:
            index_set.add(idx)
            block = blocks[idx]
            for param in block.parameters():
                param.requires_grad = trainable


def _merge_dict(base, overrides):
    merged = dict(base)
    if overrides:
        merged.update(overrides)
    return merged


def _gather_module_params(model, module_names, seen_params):
    params = []
    for module_name in module_names:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)
            params.append(param)
    return params


def _build_optimizer_param_groups(model, head_cfg):
    head_modules = head_cfg.get("head_modules") or ["pyramid", "proposal_generator", "roi_heads"]
    freeze_modules = head_cfg.get("freeze_modules") or ["preprocessing", "embedding", "backbone"]

    param_groups = []
    seen = set()

    head_params = _gather_module_params(model, head_modules, seen)
    if head_params:
        param_groups.append({"name": "head", "params": head_params})

    backbone_module_names = []
    for module_name in freeze_modules:
        if module_name not in backbone_module_names:
            backbone_module_names.append(module_name)
    backbone_params = _gather_module_params(model, backbone_module_names, seen)
    if backbone_params:
        param_groups.append({"name": "backbone", "params": backbone_params})

    remaining = [param for param in model.parameters() if id(param) not in seen]
    if remaining:
        param_groups.append({"name": "other", "params": remaining})

    return param_groups


def _configure_optimizer_for_stage(optimizer, stage, base_optimizer_kwargs):
    overrides = stage.get("optimizer_overrides") or {}
    base_lr = overrides.get("lr", base_optimizer_kwargs.get("lr", 0.0))
    base_wd = overrides.get("weight_decay", base_optimizer_kwargs.get("weight_decay", 0.0))

    head_lr = stage.get("head_lr", overrides.get("lr", base_lr))
    head_wd = stage.get("head_weight_decay", overrides.get("weight_decay", base_wd))
    backbone_lr = stage.get("backbone_lr", base_lr)
    backbone_wd = stage.get("backbone_weight_decay", base_wd)
    other_lr = stage.get("other_lr", base_lr)
    other_wd = stage.get("other_weight_decay", base_wd)

    def _lr_scale(target_lr):
        if base_lr == 0:
            return 1.0
        return target_lr / base_lr

    for group in optimizer.param_groups:
        name = group.get("name")
        if name == "head":
            group["lr_scale"] = _lr_scale(head_lr)
            group["weight_decay"] = head_wd
        elif name == "backbone":
            group["lr_scale"] = _lr_scale(backbone_lr)
            group["weight_decay"] = backbone_wd
        else:
            group["lr_scale"] = _lr_scale(other_lr)
            group["weight_decay"] = other_wd
        group["lr"] = base_lr * group.get("lr_scale", 1.0)

    return base_lr if base_lr != 0 else base_optimizer_kwargs.get("lr", 0.0)


def _build_training_stages(config, head_cfg):
    freeze_modules = head_cfg.get("freeze_modules") or ["preprocessing", "embedding", "backbone"]
    backbone_extra_modules = [module for module in freeze_modules if module != "backbone"]
    stages_cfg = head_cfg.get("stages")
    total_epochs = int(config.get("epochs", 0))
    stages = []
    consumed_epochs = 0

    def _append_stage(stage_cfg, default_name):
        nonlocal consumed_epochs
        max_epochs = max(0, total_epochs - consumed_epochs)
        stage_epochs = int(stage_cfg.get("epochs", 0) or 0)
        stage_epochs = max(0, min(stage_epochs, max_epochs))
        if stage_epochs == 0:
            return

        optimizer_cfg = stage_cfg.get("optimizer") or {}
        backbone_opt_cfg = stage_cfg.get("backbone") or {}
        lr_cfg = stage_cfg.get("lr_scheduler") or {}

        stage = {
            "name": stage_cfg.get("name", default_name),
            "type": stage_cfg.get("name", default_name),
            "description": stage_cfg.get("description"),
            "epochs": stage_epochs,
            "train_head": stage_cfg.get("train_head", True),
            "train_backbone": stage_cfg.get("train_backbone", False),
            "backbone_blocks": stage_cfg.get("backbone_blocks"),
            "backbone_extra_modules": backbone_extra_modules,
            "freeze_modules": freeze_modules,
            "optimizer_overrides": optimizer_cfg,
            "head_lr": optimizer_cfg.get("lr"),
            "head_weight_decay": optimizer_cfg.get("weight_decay"),
            "backbone_lr": backbone_opt_cfg.get("lr"),
            "backbone_weight_decay": backbone_opt_cfg.get("weight_decay"),
            "lr_overrides": lr_cfg,
        }
        stages.append(stage)
        consumed_epochs += stage_epochs

    if stages_cfg:
        for idx, stage_cfg in enumerate(stages_cfg):
            _append_stage(stage_cfg, f"stage_{idx + 1}")
            if consumed_epochs >= total_epochs:
                break
    else:
        # backward compatibility with legacy keys
        legacy_cfg = {
            "name": "head",
            "epochs": head_cfg.get("head_epochs", 0),
            "train_head": True,
            "train_backbone": False,
            "optimizer": head_cfg.get("head_optimizer_kwargs"),
            "lr_scheduler": head_cfg.get("head_lr_scheduler_kwargs"),
        }
        _append_stage(legacy_cfg, "head")
        partial_cfg = {
            "name": "partial",
            "epochs": head_cfg.get("partial_epochs", 0),
            "train_head": True,
            "train_backbone": True,
            "backbone_blocks": head_cfg.get("partial_backbone_blocks"),
            "optimizer": head_cfg.get("partial_optimizer_kwargs"),
            "backbone": {
                "lr": head_cfg.get("partial_backbone_lr"),
                "weight_decay": head_cfg.get("partial_backbone_weight_decay"),
            },
            "lr_scheduler": head_cfg.get("partial_lr_scheduler_kwargs"),
        }
        _append_stage(partial_cfg, "partial")
        full_cfg = {
            "name": "full",
            "epochs": total_epochs - consumed_epochs,
            "train_head": True,
            "train_backbone": True,
            "backbone_blocks": "all",
            "optimizer": head_cfg.get("full_optimizer_kwargs"),
            "backbone": {
                "lr": head_cfg.get("full_backbone_lr"),
                "weight_decay": head_cfg.get("full_backbone_weight_decay"),
            },
            "lr_scheduler": head_cfg.get("full_lr_scheduler_kwargs"),
        }
        _append_stage(full_cfg, "full")

    if consumed_epochs < total_epochs and stages:
        stages[-1]["epochs"] += total_epochs - consumed_epochs

    return stages


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


def train_pass(
    config, device, stage_name, local_epoch, global_epoch, model, optimizer, lr_sched, data, tensorboard, scaler
):
    model.train()
    n_items = config.get("n_items", len(data))
    accum_iter = config["accum_iter"]
    frames_processed = 0
    total_frames = _estimate_total_frames(data, n_items)
    num_workers = config.get("dataloader_workers", 8)
    pin_memory = isinstance(device, str) and device.startswith("cuda")
    amp_enabled = scaler.is_enabled() if scaler is not None else False
    optimizer.zero_grad()

    for vid_idx, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_loader = DataLoader(
            vid_item,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        video_loss = 0.0
        video_frames = 0
        for frame, annotations in vid_loader:
            frames_processed += 1
            video_frames += 1
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
                epoch_position = (local_epoch - 1) + progress
                lr_sched.adjust_learning_rate(epoch_position)

            with EventStorage():
                with autocast(enabled=amp_enabled):
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
            loss_value = loss.item()
            video_loss += loss_value
            scaler.scale(loss).backward(retain_graph=True)

            if frames_processed % accum_iter == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if tensorboard is not None:
                tensorboard.add_scalar("train/loss", loss.item(), global_step=frames_processed)

        avg_video_loss = video_loss / max(video_frames, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Train][Stage {stage_name}] Epoch {global_epoch} "
            f"Video {vid_idx + 1}/{n_items} frames={video_frames} "
            f"loss={avg_video_loss:.6f} lr={current_lr:.6e}"
        )


def val_pass(device, model, data, config):
    model.counting()
    model.clear_counts()
    model.eval()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))
    num_workers = config.get("dataloader_workers", 8)
    pin_memory = isinstance(device, str) and device.startswith("cuda")

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_loader = DataLoader(
            vid_item,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
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
    head_cfg = config.get("head_training") or {}
    stages = _build_training_stages(config, head_cfg)
    total_epochs = sum(stage["epochs"] for stage in stages)
    if total_epochs == 0:
        print("配置的训练 epoch 数为 0，跳过训练。")
        return
    if "tensorboard" in config:
        base_name = config["tensorboard"]
        now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tensorboard = SummaryWriter(f"{base_name}_{now_str}")
    else:
        tensorboard = None

    optimizer_class = getattr(optim, config["optimizer"])
    param_groups = _build_optimizer_param_groups(model, head_cfg)
    optimizer = optimizer_class(param_groups, **config["optimizer_kwargs"])
    use_amp = torch.cuda.is_available() and isinstance(device, str) and device.startswith("cuda")
    completed_epochs = 0
    for stage_idx, stage in enumerate(stages, start=1):
        print(f"\n[Stage {stage_idx}] {stage['name']} 阶段，训练 {stage['epochs']} 个 epoch")
        freeze_modules = stage.get("freeze_modules") or []
        non_backbone_modules = [module for module in freeze_modules if module != "backbone"]
        _set_modules_trainable(model, non_backbone_modules, stage.get("train_backbone", False))
        if "backbone" in freeze_modules:
            if stage.get("train_backbone"):
                backbone_blocks = stage.get("backbone_blocks")
                if backbone_blocks and backbone_blocks != "all":
                    _set_modules_trainable(model, ["backbone"], False)
                    _set_backbone_blocks_trainable(model, backbone_blocks, True)
                else:
                    _set_modules_trainable(model, ["backbone"], True)
            else:
                _set_modules_trainable(model, ["backbone"], False)

        stage_base_lr = _configure_optimizer_for_stage(optimizer, stage, config["optimizer_kwargs"])
        lr_sched_kwargs = _merge_dict(config["lr_scheduler_kwargs"], stage["lr_overrides"])
        lr_sched = LR_Scheduler(
            optimizer,
            lr_sched_kwargs["warmup_epochs"],
            lr_sched_kwargs["min_lr"],
            stage_base_lr,
            stage["epochs"],
        )
        scaler = GradScaler(enabled=use_amp)

        for local_epoch in range(stage["epochs"]):
            global_epoch = completed_epochs + local_epoch + 1
            print(f"\nEpoch {global_epoch}/{total_epochs} (stage: {stage['name']})")
            train_pass(
                config,
                device,
                stage["name"],
                local_epoch + 1,
                global_epoch,
                model,
                optimizer,
                lr_sched,
                train_data,
                tensorboard,
                scaler,
            )
            results = val_pass(device, model, val_data, config)
            model.reset()

            if isinstance(results, dict):
                for key, val in results.items():
                    tee_print(key.capitalize(), output_file)
                    tee_print(dict_string(val), output_file)
            else:
                tee_print(results, output_file)
            tee_print("", output_file)

            weight_path = Path(config["_output"]) / f"weights_epoch_{global_epoch}.pth"
            torch.save(model.state_dict(), weight_path)
            print(f"Saved weights to {weight_path}")

        completed_epochs += stage["epochs"]

    if tensorboard is not None:
        tensorboard.close()

    final_path = Path(config["_output"]) / "weights_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved weights to {final_path}")


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
    print(msg)
    model = model.to(device)

    if config.get("evaluate", False):
        print("Evaluating....")
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