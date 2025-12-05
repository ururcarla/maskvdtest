from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from datasets.argoverse import ArgoverseVID
from datasets.vid import VID, VIDResize


def build_video_dataset(
    config: Dict[str, Any],
    split: str,
    long_edge: int,
    *,
    shuffle: bool = True,
):
    """Factory that builds the requested video dataset."""
    dataset_cfg = config.get("dataset") or {}
    dataset_name = (dataset_cfg.get("name") or "vid").lower()

    resize_cfg = dataset_cfg.get("resize") or {}
    max_size = resize_cfg.get("max_size") or long_edge
    short_edge = resize_cfg.get("short_edge_length")
    if short_edge is None:
        short_edge = 640 * max_size // 1024
    transform = VIDResize(short_edge_length=short_edge, max_size=max_size)

    dataset_args = (dataset_cfg.get("args") or {}).get(dataset_name, {})

    if dataset_name == "vid":
        return _build_vid_dataset(dataset_args, split, transform, shuffle)
    if dataset_name == "argoverse":
        return _build_argoverse_dataset(dataset_args, split, transform, shuffle)

    raise ValueError(f"Unsupported dataset '{dataset_name}'.")


def _build_vid_dataset(args: Dict[str, Any], split: str, transform, shuffle: bool):
    root = Path(args.get("root", "/data1/vid_data"))
    tar_path = args.get("tar_path", "/data1/vid_data.tar")
    tar_path = Path(tar_path) if tar_path is not None else None

    return VID(
        root,
        split=split,
        tar_path=tar_path,
        shuffle=shuffle,
        combined_transform=transform,
    )


def _build_argoverse_dataset(args: Dict[str, Any], split: str, transform, shuffle: bool):
    img_root = args.get("img_root")
    ann_files = args.get("ann_files") or {}
    ann_file = ann_files.get(split)

    if img_root is None or ann_file is None:
        raise ValueError(
            "Argoverse dataset requires 'img_root' and 'ann_files.<split>' entries."
        )

    return ArgoverseVID(
        img_root=Path(img_root),
        ann_file=Path(ann_file),
        shuffle=shuffle,
        combined_transform=transform,
        min_box_size=args.get("min_box_size", 2),
    )

