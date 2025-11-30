import argparse
import json
import math
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a heatmap that only counts newly appearing objects "
            "based on dataset ground-truth annotations."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./data1", "vid_data"),
        help="Root folder that contains split folders such as vid_train / vid_val.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="vid_val",
        help="Dataset split to process (e.g. vid_train, vid_val).",
    )
    parser.add_argument(
        "--reference-height",
        type=int,
        default=None,
        help="Optional canvas height. Defaults to the max image height in the split.",
    )
    parser.add_argument(
        "--reference-width",
        type=int,
        default=None,
        help="Optional canvas width. Defaults to the max image width in the split.",
    )
    parser.add_argument(
        "--region-size",
        type=int,
        default=16,
        help="Patch (region) size used to aggregate the accumulated heatmap.",
    )
    parser.add_argument(
        "--keep-rate",
        type=float,
        default=0.4,
        help="Static keep rate used to select the top-k informative regions.",
    )
    parser.add_argument(
        "--output-npy",
        type=Path,
        default=Path("heatmap_vid_new_objects.npy"),
        help="Destination path for the pixel-wise accumulated heatmap (.npy).",
    )
    parser.add_argument(
        "--output-fig",
        type=Path,
        default=Path("heatmap_vid_new_objects.png"),
        help="Destination path for the rendered heatmap figure (.png).",
    )
    parser.add_argument(
        "--output-mask",
        type=Path,
        default=Path("heatmap_vid_new_objects_topk.npy"),
        help="Destination path for saving the selected region indices.",
    )
    return parser.parse_args()


def safe_int(token: str) -> int:
    token = token.split(".")[0]
    digits = "".join(ch for ch in token if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse frame index from token: {token}")
    return int(digits)


def parse_video_and_frame(image_meta: Dict) -> Tuple[str, int]:
    if "video_id" in image_meta and image_meta["video_id"] is not None:
        video_id = str(image_meta["video_id"])
    else:
        path = PurePosixPath(image_meta["file_name"])
        if path.parent != PurePosixPath("."):
            video_id = path.parent.name
        else:
            pieces = path.stem.split("_")
            video_id = "_".join(pieces[:-1]) if len(pieces) > 1 else pieces[0]
    if "frame_id" in image_meta and image_meta["frame_id"] is not None:
        frame_index = int(image_meta["frame_id"])
    else:
        path = PurePosixPath(image_meta["file_name"])
        frame_index = safe_int(path.stem)
    return video_id, frame_index


def select_new_objects(annotations: Iterable[Dict], images: Dict[int, Dict]):
    track_first_frame = {}
    for ann in tqdm(annotations, desc="索引GT轨迹"):
        track_id = ann.get("track_id")
        if track_id is None:
            continue
        image_meta = images[ann["image_id"]]
        video_id, frame_idx = parse_video_and_frame(image_meta)
        key = (video_id, track_id)
        if key not in track_first_frame or frame_idx < track_first_frame[key]["frame_idx"]:
            track_first_frame[key] = {
                "bbox": ann["bbox"],
                "frame_idx": frame_idx,
                "width": image_meta["width"],
                "height": image_meta["height"],
            }
    return list(track_first_frame.values())


def accumulate_heatmap(records, canvas_height: int, canvas_width: int):
    heatmap = np.zeros((canvas_height, canvas_width), dtype=np.float64)
    for record in tqdm(records, desc="累计新目标热图"):
        x, y, w, h = record["bbox"]
        scale_x = canvas_width / record["width"]
        scale_y = canvas_height / record["height"]
        x1 = max(0, min(canvas_width, int(math.floor(x * scale_x))))
        y1 = max(0, min(canvas_height, int(math.floor(y * scale_y))))
        x2 = max(0, min(canvas_width, int(math.ceil((x + w) * scale_x))))
        y2 = max(0, min(canvas_height, int(math.ceil((y + h) * scale_y))))
        if x2 <= x1 or y2 <= y1:
            continue
        heatmap[y1:y2, x1:x2] += 1.0
    return heatmap


def aggregate_regions(heatmap: np.ndarray, region: int):
    pad_h = int(math.ceil(heatmap.shape[0] / region) * region)
    pad_w = int(math.ceil(heatmap.shape[1] / region) * region)
    padded = np.zeros((pad_h, pad_w), dtype=heatmap.dtype)
    padded[: heatmap.shape[0], : heatmap.shape[1]] = heatmap
    reshaped = padded.reshape(pad_h // region, region, pad_w // region, region)
    region_scores = reshaped.sum(axis=(1, 3))
    return region_scores


def select_topk_indices(region_scores: np.ndarray, keep_rate: float):
    flat = region_scores.flatten()
    keep = max(1, int(round(keep_rate * flat.size)))
    if keep >= flat.size:
        return np.arange(flat.size, dtype=np.int64)
    threshold_index = np.argpartition(flat, -keep)[-keep:]
    return np.sort(threshold_index.astype(np.int64))


def render_heatmap(heatmap: np.ndarray, region_scores: np.ndarray, fig_path: Path, region: int):
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, cmap="magma")
    plt.title("New Object Accumulated Heatmap")
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    region_fig_path = fig_path.with_name(fig_path.stem + "_regions" + fig_path.suffix)
    plt.figure(figsize=(10, 6))
    plt.imshow(region_scores, cmap="magma")
    plt.title("Region-wise Scores")
    plt.colorbar(label=f"Accumulated count per {region}x{region} patch")
    plt.tight_layout()
    plt.savefig(region_fig_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    labels_path = args.dataset_root / args.split / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"未找到标注文件: {labels_path}")

    with labels_path.open("r") as fp:
        json_data = json.load(fp)

    images = {image["id"]: image for image in json_data["images"]}
    heights = [img["height"] for img in images.values()]
    widths = [img["width"] for img in images.values()]
    canvas_height = args.reference_height or max(heights)
    canvas_width = args.reference_width or max(widths)

    new_object_records = select_new_objects(json_data["annotations"], images)
    print(f"识别到 {len(new_object_records)} 条新目标轨迹。")
    heatmap = accumulate_heatmap(new_object_records, canvas_height, canvas_width)

    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, heatmap)
    print(f"像素级热图已保存至 {args.output_npy.resolve()}")

    region_scores = aggregate_regions(heatmap, args.region_size)
    topk_indices = select_topk_indices(region_scores, args.keep_rate)

    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_mask, topk_indices)
    print(
        f"根据 keep-rate={args.keep_rate:.2f} 选出的 {len(topk_indices)} 个区域索引已保存至 {args.output_mask.resolve()}"
    )

    if args.output_fig:
        render_heatmap(heatmap, region_scores, args.output_fig, args.region_size)
        print(f"可视化热图已输出至 {args.output_fig.resolve()} 及对应 region 版本。")


if __name__ == "__main__":
    main()

