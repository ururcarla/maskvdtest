import re
import subprocess
from collections import defaultdict
from pathlib import Path
from random import Random

import requests
import torch

from backbones.modules import SimpleSTGTGate, TokenDeltaGate, TokenGate
from util.box_ops import box_iou


class MeanValue:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def compute(self):
        return 0.0 if (self.count == 0) else self.sum / self.count

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value):
        self.sum += value
        self.count += 1


class TopKAccuracy:
    def __init__(self, k):
        self.k = k
        self.correct = 0
        self.total = 0

    def compute(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, pred, true):
        _, top_k = pred.topk(self.k, dim=-1)
        self.correct += true.eq(top_k).sum().item()
        self.total += true.numel()


def decode_video(
    input_path,
    output_path,
    name_format="%d",
    image_format="png",
    ffmpeg_input_args=None,
    ffmpeg_output_args=None,
):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    if ffmpeg_input_args is None:
        ffmpeg_input_args = []
    if ffmpeg_output_args is None:
        ffmpeg_output_args = []
    return subprocess.call(
        ["ffmpeg", "-loglevel", "error"]
        + ffmpeg_input_args
        + ["-i", input_path]
        + ffmpeg_output_args
        + [output_path / f"{name_format}.{image_format}"]
    )


def dict_to_device(x, device):
    return {key: value.to(device) for key, value in x.items()}


# https://gist.github.com/wasi0013/ab73f314f8070951b92f6670f68b2d80
def download_file(url, output_path, chunk_size=4096, verbose=True):
    if verbose:
        print(f"Downloading {url}...", flush=True)
    with requests.get(url, stream=True) as source:
        with open(output_path, "wb") as output_file:
            for chunk in source.iter_content(chunk_size=chunk_size):
                if chunk:
                    output_file.write(chunk)


def get_device_description(device):
    if device == "cuda":
        return torch.cuda.get_device_name()
    else:
        return f"CPU with {torch.get_num_threads()} threads"


def get_pytorch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_patterns(pattern_file):
    patterns = []
    last_regex = None
    with open(pattern_file, "r") as text:
        for line in text:
            line = line.strip()
            if line == "":
                continue
            elif last_regex is None:
                last_regex = re.compile(line)
            else:
                patterns.append((last_regex, line))
                last_regex = None
    return patterns


def remap_weights(in_weights, patterns, verbose=False):
    n_remapped = 0
    out_weights = {}
    for in_key, weight in in_weights.items():
        out_key = in_key
        discard = False
        for regex, replacement in patterns:
            out_key, n_matches = regex.subn(replacement, out_key)
            if n_matches > 0:
                if replacement == "DISCARD":
                    discard = True
                    out_key = "DISCARD"
                n_remapped += 1
                if verbose:
                    print(f"{in_key}  ==>  {out_key}")
                break
        if not discard:
            out_weights[out_key] = weight
    return out_weights, n_remapped


def seeded_shuffle(sequence, seed):
    rng = Random()
    rng.seed(seed)
    rng.shuffle(sequence)


def set_policies(model, policy_class, **policy_kwargs):
    for gate_class in [SimpleSTGTGate, TokenDeltaGate, TokenGate]:
        for gate in model.modules_of_type(gate_class):
            gate.policy = policy_class(**policy_kwargs)


def squeeze_dict(x, dim=None):
    return {key: value.squeeze(dim=dim) for key, value in x.items()}


def tee_print(s, file, flush=True):
    print(s, flush=flush)
    print(s, file=file, flush=flush)


def compute_detection_recall(predictions, targets, iou_threshold=0.5):
    """
    Compute overall and macro recall for detection outputs.

    :param predictions: iterable of dicts with "boxes", "scores", "labels"
    :param targets: iterable of dicts with "boxes" and "labels"
    :param iou_threshold: IoU threshold used to consider a prediction a hit
    :return: dict containing overall and macro recall plus bookkeeping stats
    """

    def _greedy_match(iou_matrix, threshold):
        matches = 0
        matched_gt = set()
        if iou_matrix.numel() == 0:
            return matches
        for row in range(iou_matrix.shape[0]):
            values = iou_matrix[row]
            if values.numel() == 0:
                continue
            iou_val, gt_idx = values.max(dim=0)
            if iou_val >= threshold:
                gt_id = int(gt_idx.item())
                if gt_id not in matched_gt:
                    matched_gt.add(gt_id)
                    matches += 1
        return matches

    recall_data = defaultdict(lambda: {"matched": 0, "total": 0})
    matched_total = 0
    total_gt = 0
    n_samples = min(len(predictions), len(targets))

    for idx in range(n_samples):
        pred = predictions[idx]
        target = targets[idx]

        gt_boxes = target.get("boxes")
        gt_labels = target.get("labels")
        if gt_boxes is None or gt_labels is None:
            continue
        gt_boxes = gt_boxes.detach().cpu()
        gt_labels = gt_labels.detach().cpu()
        total_gt += int(gt_labels.shape[0])
        for class_id in gt_labels.tolist():
            recall_data[int(class_id)]["total"] += 1
        if gt_boxes.numel() == 0:
            continue

        pred_boxes = pred.get("boxes")
        pred_labels = pred.get("labels")
        if pred_boxes is None or pred_labels is None or pred_boxes.numel() == 0:
            continue
        scores = pred.get("scores")
        pred_boxes = pred_boxes.detach().cpu()
        pred_labels = pred_labels.detach().cpu()
        if scores is not None:
            scores = scores.detach().cpu()
        if scores is not None and scores.numel() > 0:
            order = torch.argsort(scores, descending=True)
            pred_boxes = pred_boxes[order]
            pred_labels = pred_labels[order]

        iou_matrix, _ = box_iou(pred_boxes, gt_boxes)
        unique_classes = torch.unique(gt_labels)
        for cls_val in unique_classes.tolist():
            cls_val = int(cls_val)
            gt_mask = gt_labels == cls_val
            pred_mask = pred_labels == cls_val
            gt_indices = torch.nonzero(gt_mask, as_tuple=False).flatten()
            pred_indices = torch.nonzero(pred_mask, as_tuple=False).flatten()
            if gt_indices.numel() == 0 or pred_indices.numel() == 0:
                continue
            cls_iou = iou_matrix[pred_indices][:, gt_indices]
            matches = _greedy_match(cls_iou, iou_threshold)
            recall_data[cls_val]["matched"] += matches
            matched_total += matches

    if total_gt == 0:
        return {
            "recall_overall": 0.0,
            "recall_macro": 0.0,
            "gt_total": 0.0,
            "gt_matched": 0.0,
        }

    valid_classes = [stats for stats in recall_data.values() if stats["total"] > 0]
    macro_recall = (
        sum(stats["matched"] / stats["total"] for stats in valid_classes) / len(valid_classes)
        if valid_classes
        else 0.0
    )

    return {
        "recall_overall": matched_total / total_gt,
        "recall_macro": macro_recall,
        "gt_total": float(total_gt),
        "gt_matched": float(matched_total),
    }
