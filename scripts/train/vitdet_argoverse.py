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
from utils.misc import dict_to_device, squeeze_dict
from utils.misc import tee_print
from detectron2.utils.events import EventStorage
from detectron2.data.detection_utils import annotations_to_instances, BoxMode
from backbones.base import dict_string
from util.lr_sched import LR_Scheduler
from datetime import datetime
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils.misc import dict_to_device, squeeze_dict, get_pytorch_device, set_policies
from backbones.policies import TokenNormTopK


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])
    return tuple(batch)

def train_pass(config, device, epoch, model, optimizer, lr_sched, data, tensorboard, output_file):
    model.train()
    n_items = config.get("n_items", len(data))
    step = 0
    total_loss = 0
    accum_iter = config["accum_iter"]
    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        for frame, annotations in vid_item:
            step += 1
            annotation_list = []
            gt_instances = []
            for annotation in annotations:
                for i, bbox in enumerate(annotation["boxes"]):
                    annotation_dict = {
                        "bbox": bbox,
                        "category_id": annotation["labels"][i],
                        "bbox_mode": BoxMode.XYXY_ABS,
                    }
                    annotation_list.append(annotation_dict)
                gt_instance = annotations_to_instances(
                    annotation_list, frame.shape[-2:], frame.shape[-2:]
                )
                gt_instances.append(gt_instance.to(device))

            if step % accum_iter == 0:
                lr_sched.adjust_learning_rate(step / n_items + epoch)

            with EventStorage() as storage:
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
                proposals, proposal_losses = model.proposal_generator(
                    images, x, gt_instances
                )
                _, detector_losses = model.roi_heads(images, x, proposals, gt_instances)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            loss = sum(losses.values())
            loss.backward(retain_graph=True)
            total_loss += loss.item()
            if step % accum_iter == 0:
                tee_print(
                    f"Loss: {total_loss/step}, lr: {optimizer.param_groups[0]['lr']}",
                    output_file,
                )
                optimizer.step()
                optimizer.zero_grad()
            if tensorboard is not None:
                tensorboard.add_scalar("train/loss", loss.item())


def val_pass(device, model, data, config):
    model.counting()
    model.clear_counts()
    model.eval()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))
    count = 0

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        n_frames += len(vid_item)
        model.reset()
        for frame, annotations in vid_item:
            with torch.inference_mode():
                results = model(frame.to(device))
                outputs.extend(results)
            labels.extend([dict_to_device(annotation, device) for annotation in annotations])

    # for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
    #     count += 1
    #     vid_item = DataLoader(vid_item, batch_size=1)
    #     n_frames += len(vid_item)
    #     model.reset()
    #     #print(vid_item)
    #     #print("##########################")
    #     for frame, annotations in vid_item:
    #         # print(f'{count}###')
    #         # count += 1
    #         #print(frame)
    #         #print(annotations)
    #         #print("##########################")
    #         #break
    #         with torch.inference_mode():
    #             outputs.extend(model(frame.to(device)))
    #         labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))
        #break
    # MeanAveragePrecision is extremely slow. It seems fastest to call
    # update() and compute() just once, after all predictions are done.
    counts = model.total_counts() / n_frames

    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def train_vitdet(config, model, device, train_data, val_data, output_file):
    # Set up the optimizer.
    optimizer_class = getattr(optim, config["optimizer"])
    optimizer = optimizer_class(model.parameters(), **config["optimizer_kwargs"])

    lr_sched = LR_Scheduler(optimizer, config["lr_scheduler_kwargs"]["warmup_epochs"], config["lr_scheduler_kwargs"]["min_lr"], config["optimizer_kwargs"]["lr"], config["epochs"])
    # Set up TensorBoard logging.
    if "tensorboard" in config:
        base_name = config["tensorboard"]
        now_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        tensorboard = SummaryWriter(f"{base_name}_{now_str}")
    else:
        tensorboard = None

    n_epochs = config["epochs"]
    for epoch in range(n_epochs):
        tee_print(f"\nEpoch {epoch + 1}/{n_epochs}", flush=True, file=output_file)
        train_pass(config, device, epoch + 1, model, optimizer, lr_sched, train_data, tensorboard, output_file)
        results = val_pass(device, model, val_data, config)
       
        # Print and save results.
        if isinstance(results, dict):
            for key, val in results.items():
                tee_print(key.capitalize(), output_file)
                tee_print(dict_string(val), output_file)
        else:
            tee_print(results, output_file)
        tee_print("", output_file) 

        weight_path = config["_output"] + f"/weights_epoch_{epoch}.pth"
        torch.save(model.state_dict(), weight_path)
        tee_print(f"Saved weights to {weight_path}", flush=True, file=output_file)
        
    if tensorboard is not None:
        tensorboard.close()

    # Save the final weights.
    # weight_path = config["_output"] + "weights_final.pth"
    # torch.save(model.state_dict(), weight_path)
    # tee_print(f"Saved weights to {weight_path}", flush=True, file=output_file)


def main():
    config = initialize_run(config_location=Path("configs", "train", "vitdet_argoverse"))
    long_edge = max(config["model"]["input_shape"][-2:])
    n_classes = config["nb_classes"]
    dataset_cfg = config.get("dataset", {})
    dataset_root = Path(dataset_cfg.get("root", "./data/argoversehd"))
    train_split = dataset_cfg.get("train_split", "train")
    val_split = dataset_cfg.get("split", "val")
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
            verify_frames=False,
        )

    train_data = build_dataset(train_split, train_ann)
    val_data = build_dataset(val_split, val_ann)
    
    # run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)
    output_dir = Path(config["_output"])
    output_file = open(output_dir / "output.txt", 'a')

    device = get_pytorch_device()
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    # Load and set up the model.
    config["model"]["classes"] = n_classes
    model = ViTDet(**(config["model"]))
    ckpt = torch.load(config["weights"])
    for key in list(ckpt.keys()):
        if key.startswith("roi_heads.box_predictor.cls_score") or key.startswith(
            "roi_heads.box_predictor.bbox_pred"
        ):
            del ckpt[key]

    msg = model.load_state_dict(ckpt, strict=False)
    tee_print(msg, output_file)

    # TODO: map backbone weights to timm ViT model
    model = model.to(device)

    if config.get("evaluate", False):
        tee_print("Evaluating....", output_file)
        results = val_pass(device, model, val_data, config)
       
        # Print and save results.
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