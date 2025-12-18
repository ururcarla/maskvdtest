"""
Waymo Open Dataset (perception) camera → ImageNet-VID / COCO converter.

Reads camera_image + camera_box parquet shards and writes:
- ImageNet-VID style frames + labels.json (compatible with datasets.vid.VID)
- COCO video style json with seq_dirs (compatible with datasets.argoverse.ArgoverseVID)

Usage example:
python scripts/convert/waymo_to_vid.py ^
  --input-root /path/to/waymo ^
  --output-root ./data/waymo_vid ^
  --split training ^
  --export-vid --export-coco ^
  --stride 1
"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow.parquet as pq
from PIL import Image


# Waymo camera enum → human readable name
CAMERA_NAME_MAP: Dict[int, str] = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}
ALLOWED_CAMERAS = set(CAMERA_NAME_MAP.keys())

# 2D检测三类：vehicle / pedestrian / cyclist（保持与官方教程一致）
WAYMO_CLASSES: Dict[int, str] = {
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Waymo root containing training/validation parquet folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root for converted data.",
    )
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="training",
        help="Which split to convert.",
    )
    parser.add_argument(
        "--export-vid",
        action="store_true",
        help="Export ImageNet-VID style frames + labels.json.",
    )
    parser.add_argument(
        "--export-coco",
        action="store_true",
        help="Export COCO (video) json with seq_dirs.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame subsampling stride (keep every Nth frame).",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Optional cap per video to speed up debugging.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Parquet batch size during iteration.",
    )
    return parser.parse_args()


def iter_parquet_rows(path: Path, columns: List[str] | None, batch_size: int):
    """Yield rows as python dicts from a parquet file."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
        for row in batch.to_pylist():
            yield row


def get_first(row: dict, names: Iterable[str]):
    """Return first non-None value among candidate field names."""
    for name in names:
        if name is None:
            continue
        if name in row and row[name] is not None:
            return row[name]
    return None


def discover_parquet_files(directory: Path) -> List[Path]:
    return sorted(p for p in directory.glob("*.parquet"))


def bbox_from_row(row: dict) -> Tuple[float, float, float, float] | None:
    """Extract xyxy bbox from a camera_box row with multiple schema fallbacks."""
    # Direct xyxy (common parquet export)
    x_min = get_first(row, ["bbox_xmin", "bbox_x_min"])
    y_min = get_first(row, ["bbox_ymin", "bbox_y_min"])
    x_max = get_first(row, ["bbox_xmax", "bbox_x_max"])
    y_max = get_first(row, ["bbox_ymax", "bbox_y_max"])
    if None not in (x_min, y_min, x_max, y_max):
        return float(x_min), float(y_min), float(x_max), float(y_max)

    # Waymo parquet schema with flattened names
    cx = get_first(row, ["[CameraBoxComponent].box.center.x", "box.center_x", "box.center.x"])
    cy = get_first(row, ["[CameraBoxComponent].box.center.y", "box.center_y", "box.center.y"])
    w = get_first(row, ["[CameraBoxComponent].box.size.x", "box.length", "box.size.x", "box.width"])
    h = get_first(row, ["[CameraBoxComponent].box.size.y", "box.width", "box.size.y", "box.length"])
    if None in (cx, cy, w, h):
        return None
    x_min = float(cx) - float(w) / 2.0
    x_max = float(cx) + float(w) / 2.0
    y_min = float(cy) - float(h) / 2.0
    y_max = float(cy) + float(h) / 2.0
    return x_min, y_min, x_max, y_max


def build_box_index(
    parquet_paths: Iterable[Path], batch_size: int
) -> Dict[Tuple[str, int, int], List[dict]]:
    """
    Pre-index 2D boxes keyed by (segment, camera_id, timestamp).
    Stored fields: bbox_xyxy, label_id, object_id.
    """
    index: Dict[Tuple[str, int, int], List[dict]] = defaultdict(list)
    columns = [
        "key",
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        "key.camera_name",
        "key.camera_object_id",
        "box",
        "label",
        "type",
        "object_id",
        "id",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "[CameraBoxComponent].box.center.x",
        "[CameraBoxComponent].box.center.y",
        "[CameraBoxComponent].box.size.x",
        "[CameraBoxComponent].box.size.y",
        "[CameraBoxComponent].type",
    ]
    for path in parquet_paths:
        for row in iter_parquet_rows(path, columns=columns, batch_size=batch_size):
            cam_id = get_first(row, ["key.camera_name"])
            if cam_id not in ALLOWED_CAMERAS:
                continue
            segment = get_first(row, ["key.segment_context_name"])
            ts = get_first(row, ["key.frame_timestamp_micros"])
            if segment is None or ts is None:
                continue

            bbox = bbox_from_row(row)
            if bbox is None:
                continue

            label = get_first(row, ["label", "type", "[CameraBoxComponent].type"])
            if label is None or label not in WAYMO_CLASSES:
                continue

            object_id = get_first(row, ["object_id", "id", "key.camera_object_id"])
            index[(segment, cam_id, ts)].append(
                {"bbox": bbox, "label": int(label), "object_id": object_id}
            )
    return index


def normalize_video_id(segment: str, cam_name: str) -> str:
    """Use hyphenated id to keep VID parsing simple (no extra underscores)."""
    safe_segment = segment.replace("_", "-")
    return f"{safe_segment}-{cam_name.lower()}"


def clip_bbox(
    bbox: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    return x1, y1, x2, y2


def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_vid_json(
    frames: Dict[str, List[dict]], out_dir: Path, categories: Dict[int, str]
):
    """Write labels.json compatible with datasets.vid.VID."""
    images = []
    annotations = []
    ann_id = 1

    for video_id, video_frames in frames.items():
        for frame in video_frames:
            images.append(
                {
                    "id": frame["image_id"],
                    "file_name": frame["file_name"],
                    "height": frame["height"],
                    "width": frame["width"],
                }
            )
            for box, label in zip(frame["boxes_xyxy"], frame["labels"]):
                x1, y1, x2, y2 = box
                bbox = [x1, y1, x2 - x1, y2 - y1]
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": frame["image_id"],
                        "category_id": label,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    categories_list = [{"id": cid, "name": name} for cid, name in categories.items()]
    payload = {"images": images, "annotations": annotations, "categories": categories_list}
    out_file = out_dir / "labels.json"
    out_file.write_text(json.dumps(payload))
    print(f"[VID] Wrote {out_file} ({len(images)} images, {len(annotations)} boxes)")


def write_coco_json(
    seq_frames: Dict[str, List[dict]],
    out_file: Path,
    categories: Dict[int, str],
):
    """Write COCO-style json with seq_dirs for ArgoverseVID loader."""
    images = []
    annotations = []
    seq_dirs = []
    ann_id = 1
    image_id = 1

    for sid, (video_id, video_frames) in enumerate(seq_frames.items()):
        seq_dirs.append(video_id)
        for fid, frame in enumerate(video_frames):
            images.append(
                {
                    "id": image_id,
                    "sid": sid,
                    "fid": fid,
                    "name": frame["coco_name"],
                    "height": frame["height"],
                    "width": frame["width"],
                }
            )
            for box, label, track in zip(
                frame["boxes_xyxy"], frame["labels"], frame["track_ids"]
            ):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "track": track,
                    }
                )
                ann_id += 1
            image_id += 1

    categories_list = [{"id": cid, "name": name} for cid, name in categories.items()]
    payload = {
        "seq_dirs": seq_dirs,
        "images": images,
        "annotations": annotations,
        "categories": categories_list,
    }
    out_file.write_text(json.dumps(payload))
    print(
        f"[COCO] Wrote {out_file} ({len(seq_dirs)} sequences, {len(images)} images, {len(annotations)} boxes)"
    )


def ensure_export_flags(args: argparse.Namespace):
    if not args.export_vid and not args.export_coco:
        args.export_vid = True
        args.export_coco = True


def main():
    args = parse_args()
    ensure_export_flags(args)

    image_dir = args.input_root / args.split / "camera_image"
    box_dir = args.input_root / args.split / "camera_box"

    image_paths = discover_parquet_files(image_dir)
    box_paths = discover_parquet_files(box_dir)
    if len(image_paths) == 0 or len(box_paths) == 0:
        raise FileNotFoundError("camera_image or camera_box parquet files not found.")

    print(f"Indexing boxes from {len(box_paths)} shards...")
    box_index = build_box_index(box_paths, batch_size=args.batch_size)
    print(f"Indexed {sum(len(v) for v in box_index.values())} boxes.")

    out_split_dir = args.output_root / ("train" if args.split == "training" else "val")
    frames_root = out_split_dir / "frames"
    ensure_dirs(frames_root)

    vid_frames: Dict[str, List[dict]] = defaultdict(list)
    coco_frames: Dict[str, List[dict]] = defaultdict(list)

    frame_counters: Dict[str, int] = defaultdict(int)
    image_id = 1

    image_columns = [
        "key",
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        "key.camera_name",
        "image",
        "encoded_image",
        "[CameraImageComponent].image",
        "width",
        "height",
    ]
    for img_path in image_paths:
        for row in iter_parquet_rows(img_path, columns=image_columns, batch_size=args.batch_size):
            cam_id = get_first(row, ["key.camera_name"])
            if cam_id not in ALLOWED_CAMERAS:
                continue
            segment = get_first(row, ["key.segment_context_name"])
            ts = get_first(row, ["key.frame_timestamp_micros"])
            if segment is None or ts is None:
                continue

            cam_name = CAMERA_NAME_MAP.get(cam_id, f"cam{cam_id}")
            video_id = normalize_video_id(segment, cam_name)

            idx = frame_counters[video_id]
            frame_counters[video_id] += 1
            if idx % args.stride != 0:
                continue
            if args.max_frames_per_video is not None and idx >= args.max_frames_per_video:
                continue

            encoded = get_first(row, ["image", "encoded_image", "[CameraImageComponent].image"])
            if encoded is None:
                continue
            width = get_first(row, ["width"])
            height = get_first(row, ["height"])
            if width is None or height is None:
                with Image.open(io.BytesIO(encoded)) as im:
                    width, height = im.size

            boxes = box_index.get((segment, cam_id, ts), [])
            boxes_xyxy = []
            labels = []
            track_ids = []
            for box in boxes:
                clipped = clip_bbox(box["bbox"], int(width), int(height))
                x1, y1, x2, y2 = clipped
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes_xyxy.append(clipped)
                labels.append(box["label"])
                obj = box.get("object_id")
                track_ids.append(hash(obj) % 1_000_000 if obj is not None else -1)

            frame_idx = idx // args.stride
            file_basename = f"{frame_idx:06d}.jpg"
            vid_file_name = f"{video_id}/{video_id}_{frame_idx:06d}.jpg"

            video_dir = frames_root / video_id
            ensure_dirs(video_dir)
            (video_dir / file_basename).write_bytes(encoded)

            if args.export_vid:
                vid_frames[video_id].append(
                    {
                        "image_id": image_id,
                        "file_name": vid_file_name,
                        "height": int(height),
                        "width": int(width),
                        "boxes_xyxy": boxes_xyxy,
                        "labels": labels,
                    }
                )

            if args.export_coco:
                coco_frames[video_id].append(
                    {
                        "coco_name": file_basename,
                        "height": int(height),
                        "width": int(width),
                        "boxes_xyxy": boxes_xyxy,
                        "labels": labels,
                        "track_ids": track_ids,
                    }
                )
            image_id += 1

    if args.export_vid:
        write_vid_json(vid_frames, out_split_dir, WAYMO_CLASSES)
        (out_split_dir / "unpacked").touch()

    if args.export_coco:
        ann_dir = out_split_dir / "annotations"
        ensure_dirs(ann_dir)
        ann_path = ann_dir / f"instances_{'train' if args.split == 'training' else 'val'}.json"
        write_coco_json(coco_frames, ann_path, WAYMO_CLASSES)

    print("Done.")


if __name__ == "__main__":
    main()
