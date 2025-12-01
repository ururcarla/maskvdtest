import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

CATS = [
    "Pedestrian",
    "Car",
    "Cyclist",
    "Van",
    "Truck",
    "Person_sitting",
    "Tram",
    "Misc",
    "DontCare",
]

# mapping to consolidated category ids (same as the original script)
CAT_IDS = {1: 1, 2: 2, 3: 3, 4: 2, 5: 2, 6: 1, 7: 0, 8: 0, 9: 0}
CAT_INFO = [
    {
        "name": cat,
        "id": CAT_IDS[CATS.index(cat) + 1]
        if cat != "Person"
        else CAT_IDS[CATS.index("Person_sitting") + 1],
    }
    for cat in CATS
]

VIDEO_SETS = {
    "train": range(21),
    "train_half": range(21),
    "val_half": range(21),
    "val": range(21),
    "test": range(29),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert KITTI Tracking annotations (training/testing directory structure) "
            "to a COCO-style JSON with track_id preserved."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./data1/kitti_tracking"),
        help="Root directory containing training/ and testing/ folders.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train_half", "val_half"],
        choices=list(VIDEO_SETS.keys()),
        help="Splits to convert. Default uses the common train/val halves.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store tracking_<split>.json (defaults to <dataset-root>/annotations).",
    )
    parser.add_argument(
        "--create-half-labels",
        action="store_true",
        help=(
            "Optionally dump per-split KITTI text labels (label_02_<split>) "
            "similar to the original script."
        ),
    )
    return parser.parse_args()


def project_to_image(points_3d: np.ndarray, projection: np.ndarray) -> np.ndarray:
    pts_homo = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = (projection @ pts_homo.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d


def read_calibration(calib_path: Path) -> np.ndarray:
    with calib_path.open("r") as fp:
        lines = fp.readlines()
    if len(lines) < 3:
        raise ValueError(f"Calibration file {calib_path} has unexpected format.")
    calib = np.array(lines[2].strip().split(" ")[1:], dtype=np.float32).reshape(3, 4)
    return calib


def bbox_to_coco(bbox: Sequence[float]) -> List[float]:
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


def bbox_area(bbox: Sequence[float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def determine_frame_range(split: str, total_frames: int) -> Tuple[int, int]:
    if "half" not in split:
        return 0, total_frames - 1
    half = total_frames // 2
    if split == "train_half":
        return 0, max(half - 1, 0)
    return half, total_frames - 1


def iterable_video_ids(split: str) -> Iterable[int]:
    if split not in VIDEO_SETS:
        raise ValueError(f"Unsupported split {split}.")
    return VIDEO_SETS[split]


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def convert_split(split: str, args):
    subset = "testing" if split == "test" else "training"
    image_root = args.dataset_root / subset / "image_02"
    calib_root = args.dataset_root / subset / "calib"
    label_root = None if subset == "testing" else args.dataset_root / subset / "label_02"

    if not image_root.exists():
        raise FileNotFoundError(f"Missing image directory: {image_root}")
    if not calib_root.exists():
        raise FileNotFoundError(f"Missing calibration directory: {calib_root}")
    if subset == "training" and (label_root is None or not label_root.exists()):
        raise FileNotFoundError(f"Missing label directory: {label_root}")

    ret: Dict[str, List] = {"images": [], "annotations": [], "videos": [], "categories": CAT_INFO}
    image_id_lookup: Dict[Tuple[str, int], int] = {}
    image_counter = 1
    annotation_counter = 1

    for video_idx in iterable_video_ids(split):
        video_name = f"{video_idx:04d}"
        video_image_dir = image_root / video_name
        if not video_image_dir.exists():
            print(f"[WARN] Skipping missing video directory {video_image_dir}")
            continue
        frame_files = sorted(video_image_dir.glob("*.png"))
        if not frame_files:
            print(f"[WARN] No frames found in {video_image_dir}")
            continue

        frame_start, frame_end = determine_frame_range(split, len(frame_files))
        calib = read_calibration(calib_root / f"{video_name}.txt")
        ret["videos"].append({"id": video_idx + 1, "file_name": video_name})

        for local_idx, frame_path in enumerate(frame_files):
            if local_idx < frame_start or local_idx > frame_end:
                continue
            image_id_lookup[(video_name, local_idx)] = image_counter
            ret["images"].append(
                {
                    "file_name": f"{video_name}/{frame_path.name}",
                    "id": image_counter,
                    "video_id": video_idx + 1,
                    "frame_id": local_idx - frame_start + 1,
                    "calib": calib.tolist(),
                }
            )
            image_counter += 1

        if split == "test":
            continue

        label_file = label_root / f"{video_name}.txt"
        if not label_file.exists():
            print(f"[WARN] Missing label file {label_file}, skipping annotations for this video.")
            continue

        split_label_dir = args.dataset_root / subset / f"label_02_{split}"
        split_label_dir.mkdir(exist_ok=True, parents=True) if args.create_half_labels else None
        split_label_fp = (
            (split_label_dir / f"{video_name}.txt").open("w") if args.create_half_labels else None
        )

        with label_file.open("r") as fp:
            for line in fp:
                if not line.strip():
                    continue
                parts = line.strip().split(" ")
                frame_id = int(parts[0])
                track_id = int(parts[1])
                obj_type = parts[2]
                if obj_type == "Person":
                    cat_id = CAT_IDS[CATS.index("Person_sitting") + 1]
                else:
                    cat_id = CAT_IDS[CATS.index(obj_type) + 1]
                bbox = [float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])]
                dimensions = [float(parts[10]), float(parts[11]), float(parts[12])]
                location = [float(parts[13]), float(parts[14]), float(parts[15])]
                rotation_y = float(parts[16])
                image_index = frame_id

                if image_index < frame_start or image_index > frame_end:
                    continue

                if args.create_half_labels:
                    shifted_id = frame_id - frame_start
                    split_label_fp and split_label_fp.write(
                        f"{shifted_id} {' '.join(parts[1:])}\n"
                    )

                if cat_id == 0:
                    continue

                image_global_id = image_id_lookup[(video_name, image_index)]
                amodel_center = project_to_image(
                    np.array(
                        [[location[0], location[1] - dimensions[0] / 2, location[2]]], dtype=np.float32
                    ),
                    calib,
                )[0].tolist()

                annotation = {
                    "image_id": image_global_id,
                    "id": annotation_counter,
                    "category_id": cat_id,
                    "bbox": bbox_to_coco(bbox),
                    "area": bbox_area(bbox),
                    "iscrowd": 0,
                    "dim": dimensions,
                    "location": location,
                    "rotation_y": rotation_y,
                    "alpha": float(parts[5]),
                    "truncated": int(float(parts[3])),
                    "occluded": int(parts[4]),
                    "amodel_center": amodel_center,
                    "track_id": track_id + 1,
                    "depth": location[2],
                }
                ret["annotations"].append(annotation)
                annotation_counter += 1

        if split_label_fp is not None:
            split_label_fp.close()

    output_dir = args.output_dir or (args.dataset_root / "annotations")
    ensure_output_dir(output_dir)
    output_path = output_dir / f"tracking_{split}.json"
    with output_path.open("w") as fp:
        json.dump(ret, fp)
    print(
        f"[INFO] Saved split '{split}' with {len(ret['images'])} images and "
        f"{len(ret['annotations'])} annotations to {output_path}"
    )


def main():
    args = parse_args()
    for split in args.splits:
        convert_split(split, args)


if __name__ == "__main__":
    main()

