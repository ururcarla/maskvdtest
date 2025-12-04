#!/usr/bin/env python3

"""
Reorganize Argoverse-HD into a VID-style directory layout:

<output_root>/
  train/
    frames/<sequence_id>/<frame>.jpg
    labels.json
  val/
    ...

The resulting structure can be consumed directly by datasets.VID.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm

from datasets.argoverse_hd import ArgoverseHD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Argoverse-HD in VID layout")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root containing Argoverse-1.1/ and Argoverse-HD/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root to store reorganized splits",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help='Splits to export (defaults to ["train", "val"])',
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy image files instead of creating hard links",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip frames whose source image cannot be found",
    )
    return parser.parse_args()


def ensure_parent(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)


def replicate_frame(src: Path, dst: Path, force_copy: bool) -> None:
    if dst.exists():
        return
    ensure_parent(dst)
    try:
        if force_copy:
            shutil.copy2(src, dst)
        else:
            os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def tensor_to_list(tensor) -> List[float]:
    if hasattr(tensor, "tolist"):
        return tensor.tolist()
    return list(tensor)


def export_split(dataset: ArgoverseHD, output_dir: Path, force_copy: bool, allow_missing: bool) -> None:
    with dataset.annotations_path.open("r") as fp:
        original_json = json.load(fp)
    categories = original_json.get("categories", [])

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    videos = []
    image_id = 0
    annotation_id = 0

    for video in tqdm(dataset.video_info, desc=f"Export {dataset.split}", unit="video"):
        video_id = video["video_id"]
        videos.append({"id": video_id, "file_name": video_id})
        seq_dir = frames_dir / video_id
        seq_dir.mkdir(parents=True, exist_ok=True)
        for local_idx, frame in enumerate(video["frames"]):
            src_path = Path(frame["path"])
            dest_name = f"{local_idx:06d}{src_path.suffix}"
            dest_path = seq_dir / dest_name
            if not src_path.exists():
                if allow_missing:
                    continue
                raise FileNotFoundError(f"Missing frame: {src_path}")
            replicate_frame(src_path, dest_path, force_copy)
            frame_meta = frame["annotations"]
            frame_height = frame_meta.get("height")
            frame_width = frame_meta.get("width")
            if frame_height is None or frame_width is None:
                try:
                    from PIL import Image

                    with Image.open(src_path) as img:
                        frame_width, frame_height = img.size
                except Exception:
                    frame_width = frame_width or 0
                    frame_height = frame_height or 0
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{video_id}/{dest_name}",
                    "video_id": video_id,
                    "frame_id": int(local_idx),
                    "height": frame_height,
                    "width": frame_width,
                }
            )

            boxes = frame_meta.get("boxes")
            labels = frame_meta.get("labels")
            if boxes is None or labels is None:
                image_id += 1
                continue
            boxes_list = tensor_to_list(boxes)
            labels_list = tensor_to_list(labels)
            for box, label in zip(boxes_list, labels_list):
                x1, y1, x2, y2 = box
                w = max(0.0, float(x2) - float(x1))
                h = max(0.0, float(y2) - float(y1))
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(label) + 1,
                        "bbox": [float(x1), float(y1), w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
            image_id += 1

    labels_path = output_dir / "labels.json"
    with labels_path.open("w") as fp:
        json.dump(
            {
                "images": images,
                "annotations": annotations,
                "categories": categories,
                "videos": videos,
            },
            fp,
        )


def main():
    args = parse_args()
    for split in args.splits:
        dataset = ArgoverseHD(
            location=args.dataset_root,
            split=split,
            verify_frames=not args.allow_missing,
            shuffle=False,
        )
        output_split_dir = args.output_root / split
        output_split_dir.mkdir(parents=True, exist_ok=True)
        export_split(dataset, output_split_dir, force_copy=args.copy, allow_missing=args.allow_missing)


if __name__ == "__main__":
    main()

