#!/usr/bin/env python3
"""
随机抽取 ImageNet VID 连续帧窗口，基于 GT 标注生成区域热图与帧集合拼图。

使用方式：
    python scripts/misc/sample_vid_window_heatmaps.py \
        --dataset-root ./data1/vid_data \
        --split vid_val \
        --num-groups 10 \
        --max-frames 90 \
        --output-dir ./results/window_heatmaps
"""

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datasets.vid import VID  # noqa: E402


@dataclass
class FrameRecord:
    path: Path
    boxes: torch.Tensor
    frame_name: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="随机采样 VID 连续帧，生成 GT 区域热图与帧集合拼图。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./data1", "vid_data"),
        help="VID 数据根目录（包含 vid_val/vid_train 等子目录）。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="vid_val",
        help="使用的数据 split，例如 vid_val、vid_train、vid_minival。",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=10,
        help="随机抽取的连续帧窗口数量。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=90,
        help="每个窗口最多采样的连续帧数。",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="每个窗口至少需要的帧数；不足则重新采样。",
    )
    parser.add_argument(
        "--region-size",
        type=int,
        default=16,
        help="热图累计时的区域尺寸（像素，对应 stride）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="随机种子，保证可重复抽样。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results", "window_heatmaps"),
        help="输出目录，包含各组热图、帧集合图及汇总拼接图。",
    )
    parser.add_argument(
        "--collage-cols",
        type=int,
        default=5,
        help="帧集合图的列数。",
    )
    parser.add_argument(
        "--collage-rows",
        type=int,
        default=5,
        help="帧集合图的行数。",
    )
    parser.add_argument(
        "--heatmap-cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap，用于渲染区域热图。",
    )
    parser.add_argument(
        "--heatmap-mosaic-cols",
        type=int,
        default=5,
        help="最终 10 张热图拼接时的列数。",
    )
    return parser.parse_args()


def load_video_infos(dataset_root: Path, split: str):
    labels_path = dataset_root / split / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"未找到 {labels_path}，请确认数据已解压。")
    video_infos = VID._get_videos_info(dataset_root, split)  # pylint: disable=protected-access
    frames_root = dataset_root / split / "frames"
    if not frames_root.exists():
        raise FileNotFoundError(f"未找到帧目录 {frames_root}，请确认图像已就绪。")
    return video_infos, frames_root


def sample_frame_groups(
    video_infos,
    frames_root: Path,
    num_groups: int,
    max_frames: int,
    min_frames: int,
    seed: int,
) -> List[List[FrameRecord]]:
    rng = random.Random(seed)
    groups: List[List[FrameRecord]] = []
    attempts = 0
    max_attempts = num_groups * 50
    visited = set()
    min_frames = max(1, min(min_frames, max_frames))

    while len(groups) < num_groups and attempts < max_attempts:
        attempts += 1
        video_info = rng.choice(video_infos)
        frames = video_info["frames"]
        if not frames:
            continue
        start = rng.randrange(len(frames))
        end = min(len(frames), start + max_frames)
        window = frames[start:end]
        if not window:
            continue
        if len(window) < min_frames:
            continue
        key = (video_info["video_id"], window[0]["filename"])
        if key in visited:
            continue
        visited.add(key)
        frame_records = []
        for frame in window:
            frame_path = frames_root / video_info["video_id"] / frame["filename"]
            frame_records.append(
                FrameRecord(
                    path=frame_path,
                    boxes=frame["annotations"]["boxes"].detach().cpu().clone(),
                    frame_name=frame["filename"],
                )
            )
        groups.append(frame_records)
    if len(groups) < num_groups:
        print(
            f"[警告] 仅采样到 {len(groups)} 组窗口（目标 {num_groups}）。"
            "请确认数据完整、适当降低 min_frames 或增大 max_frames。",
        )
    return groups


def safe_image_size(path: Path) -> Optional[tuple]:
    if not path.exists():
        print(f"[警告] 跳过不存在的图像：{path}")
        return None
    with Image.open(path) as img:
        return img.height, img.width


def accumulate_region_heatmap(
    frames: Iterable[FrameRecord],
    region_size: int,
) -> Optional[np.ndarray]:
    size_cache = []
    prepared = []
    for record in frames:
        size = safe_image_size(record.path)
        if size is None:
            continue
        height, width = size
        boxes = record.boxes
        if boxes is None or boxes.numel() == 0:
            continue
        prepared.append((boxes.numpy(), height, width))
        size_cache.append((height, width))
    if not prepared:
        return None
    canvas_height = max(h for h, _ in size_cache)
    canvas_width = max(w for _, w in size_cache)
    grid_h = math.ceil(canvas_height / region_size)
    grid_w = math.ceil(canvas_width / region_size)
    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
    for boxes, height, width in prepared:
        scale_x = canvas_width / float(width)
        scale_y = canvas_height / float(height)
        scaled = boxes.copy()
        scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]] * scale_x, 0, canvas_width)
        scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]] * scale_y, 0, canvas_height)
        for x1, y1, x2, y2 in scaled:
            if x2 <= x1 or y2 <= y1:
                continue
            start_col = int(math.floor(x1 / region_size))
            end_col = int(math.ceil(x2 / region_size))
            start_row = int(math.floor(y1 / region_size))
            end_row = int(math.ceil(y2 / region_size))
            start_col = max(0, min(start_col, grid_w))
            end_col = max(0, min(end_col, grid_w))
            start_row = max(0, min(start_row, grid_h))
            end_row = max(0, min(end_row, grid_h))
            if end_col <= start_col or end_row <= start_row:
                continue
            heatmap[start_row:end_row, start_col:end_col] += 1.0
    return heatmap


def render_heatmap_image(
    heatmap: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str,
):
    if heatmap is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, cmap=cmap, interpolation="nearest")
    plt.title(title)
    plt.colorbar(label="GT 区域累计次数")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def sample_indices(total: int, limit: int) -> List[int]:
    if total <= limit:
        return list(range(total))
    step = total / float(limit)
    indices = [int(round(i * step)) for i in range(limit)]
    indices = [min(idx, total - 1) for idx in indices]
    return sorted(set(indices))[:limit]


def build_collage(
    frames: List[FrameRecord],
    output_path: Path,
    rows: int,
    cols: int,
):
    capacity = rows * cols
    indices = sample_indices(len(frames), capacity)
    images = []
    for idx in indices:
        record = frames[idx]
        if not record.path.exists():
            continue
        tensor = read_image(str(record.path)).float() / 255.0
        images.append(tensor)
    if not images:
        print(f"[警告] 无法为 {output_path} 构建集合图（无可用帧）。")
        return
    grid = make_grid(images, nrow=cols, padding=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(output_path))


def build_heatmap_mosaic(image_paths: List[Path], output_path: Path, cols: int):
    valid_images = []
    for path in image_paths:
        if path.exists():
            valid_images.append(path)
    if not valid_images:
        return
    tiles = [Image.open(path).convert("RGB") for path in valid_images]
    tile_w = max(img.width for img in tiles)
    tile_h = max(img.height for img in tiles)
    cols = max(1, cols)
    rows = math.ceil(len(tiles) / cols)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(0, 0, 0))
    for idx, img in enumerate(tiles):
        resized = img.resize((tile_w, tile_h), Image.BILINEAR)
        r, c = divmod(idx, cols)
        canvas.paste(resized, (c * tile_w, r * tile_h))
        img.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    video_infos, frames_root = load_video_infos(args.dataset_root, args.split)
    groups = sample_frame_groups(
        video_infos,
        frames_root,
        args.num_groups,
        args.max_frames,
        args.min_frames,
        args.seed,
    )
    heatmap_paths = []
    for idx, frames in enumerate(groups):
        heatmap = accumulate_region_heatmap(frames, args.region_size)
        heatmap_path = args.output_dir / "heatmaps" / f"group_{idx:02d}.png"
        render_heatmap_image(
            heatmap,
            heatmap_path,
            title=f"{args.split} - Group #{idx} ({len(frames)} frames)",
            cmap=args.heatmap_cmap,
        )
        heatmap_paths.append(heatmap_path)
        collage_path = args.output_dir / "collages" / f"group_{idx:02d}.png"
        build_collage(frames, collage_path, args.collage_rows, args.collage_cols)
        print(
            f"[完成] Group {idx}: {len(frames)} 帧，热图 -> {heatmap_path.name}，集合图 -> {collage_path.name}",
        )
    mosaic_path = args.output_dir / "heatmap_mosaic.png"
    build_heatmap_mosaic(heatmap_paths, mosaic_path, args.heatmap_mosaic_cols)
    print(f"热图拼接已输出到 {mosaic_path}")


if __name__ == "__main__":
    main()

