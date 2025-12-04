import argparse
import json
import math
import random
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "随机抽取若干视频片段，累计每段内不超过指定帧数的GT框，"
            "输出对应的区域热力图并生成合成大图。"
        )
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data1", "vid_data", "vid_minival", "labels.json"),
        help="COCO格式的标注文件路径（默认指向 ImageNet VID vid_minival）。",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="要输出的热力图数量（默认 10）。",
    )
    parser.add_argument(
        "--frames-per-sample",
        type=int,
        default=30,
        help="单个热力图累积的最大帧数（默认 30）。",
    )
    parser.add_argument(
        "--min-boxes",
        type=int,
        default=8,
        help="每个样本至少包含的GT框数量，低于该阈值会重新抽样（默认 8）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="随机种子，保证可复现性（默认 2024）。",
    )
    parser.add_argument(
        "--canvas-height",
        type=int,
        default=None,
        help="热力图画布高度，默认使用数据集中最大高度。",
    )
    parser.add_argument(
        "--canvas-width",
        type=int,
        default=None,
        help="热力图画布宽度，默认使用数据集中最大宽度。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs", "random_gt_heatmaps"),
        help="单张热力图的输出目录。",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=None,
        help="真实帧图像所在目录，默认尝试使用 labels.json 同级（或上级）的 frames 文件夹。",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help="10张热力图的合成输出路径（默认保存在 output-dir 下）。",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=5,
        help="合成图的列数（默认 5，即 5x2 摆放 10 张图）。",
    )
    parser.add_argument(
        "--frames-per-image",
        type=int,
        default=None,
        help="单个帧拼图包含的帧数，默认与 --frames-per-sample 相同。",
    )
    parser.add_argument(
        "--frame-output-dir",
        type=Path,
        default=None,
        help="帧拼图输出目录（默认 output-dir/frame_mosaics）。",
    )
    parser.add_argument(
        "--frame-grid-cols",
        type=int,
        default=6,
        help="帧拼图网格列数，默认 6（30 帧时约为 5x6 网格）。",
    )
    parser.add_argument(
        "--frame-thumb-max-side",
        type=int,
        default=320,
        help="帧缩略图的最大边长像素（默认 320）。",
    )
    return parser.parse_args()


def safe_int(token: str) -> int:
    token = token.split(".")[0]
    digits = "".join(ch for ch in token if ch.isdigit())
    if not digits:
        raise ValueError(f"无法从 {token} 中解析帧编号。")
    return int(digits)


def parse_video_and_frame(meta: Dict) -> Tuple[str, int]:
    if "video_id" in meta and meta["video_id"] not in (None, ""):
        video_id = str(meta["video_id"])
    else:
        path = PurePosixPath(meta["file_name"])
        if path.parent != PurePosixPath("."):
            video_id = path.parent.name
        else:
            pieces = path.stem.split("_")
            video_id = "_".join(pieces[:-1]) if len(pieces) > 1 else pieces[0]
    if "frame_id" in meta and meta["frame_id"] not in (None, ""):
        frame_idx = int(meta["frame_id"])
    else:
        frame_idx = safe_int(PurePosixPath(meta["file_name"]).stem)
    return video_id, frame_idx


def load_dataset(labels_path: Path):
    with labels_path.open("r") as fp:
        data = json.load(fp)
    images = {}
    for image in data["images"]:
        video_id, frame_idx = parse_video_and_frame(image)
        images[image["id"]] = {
            "image_id": image["id"],
            "video_id": video_id,
            "frame_idx": frame_idx,
            "width": image["width"],
            "height": image["height"],
            "file_name": image["file_name"],
            "boxes": [],
        }
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in images:
            continue
        x, y, w, h = ann["bbox"]
        images[image_id]["boxes"].append([x, y, x + w, y + h])
    videos: Dict[str, List[Dict]] = {}
    heights = []
    widths = []
    for frame in images.values():
        videos.setdefault(frame["video_id"], []).append(frame)
        heights.append(frame["height"])
        widths.append(frame["width"])
    for frames in videos.values():
        frames.sort(key=lambda item: item["frame_idx"])
    max_height = max(heights) if heights else 0
    max_width = max(widths) if widths else 0
    return videos, max_height, max_width


def resolve_frame_path(frame: Dict, frames_root: Optional[Path]) -> Optional[Path]:
    if frames_root is None:
        return None
    relative = Path(PurePosixPath(frame["file_name"]))
    video_id = str(frame["video_id"])
    frame_idx = frame["frame_idx"]
    candidates = []
    seen = set()

    def add_candidate(path: Optional[Path]):
        if path is None:
            return
        resolved = Path(path)
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    add_candidate(frames_root / relative)
    if not relative.suffix:
        for ext in (".jpg", ".jpeg", ".png"):
            add_candidate((frames_root / relative).with_suffix(ext))
    add_candidate(frames_root / relative.name)
    if not relative.name.endswith((".jpg", ".jpeg", ".png")):
        for ext in (".jpg", ".jpeg", ".png"):
            add_candidate((frames_root / relative.name).with_suffix(ext))
    add_candidate(frames_root / video_id / relative.name)
    if not relative.name.endswith((".jpg", ".jpeg", ".png")):
        for ext in (".jpg", ".jpeg", ".png"):
            add_candidate((frames_root / video_id / relative.name).with_suffix(ext))
    idx_tokens = {f"{frame_idx:06d}", f"{frame_idx:05d}", str(frame_idx)}
    for token in idx_tokens:
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            add_candidate(frames_root / video_id / f"{token}{ext}")

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def select_frame_indices(total: int, target: int) -> List[int]:
    if target <= 0 or total <= 0:
        return []
    if total <= target:
        return list(range(total))
    return list(np.linspace(0, total - 1, target, dtype=int))


def resize_with_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side is None or max_side <= 0:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return image.resize(new_size, Image.BILINEAR)


def render_frame_grid(
    image_paths: Sequence[Path],
    grid_cols: int,
    output_path: Path,
    thumb_max_side: int,
    padding: int = 8,
    margin: int = 18,
    bg_color=(12, 12, 12),
):
    if not image_paths:
        return
    thumbs: List[Image.Image] = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                thumbs.append(resize_with_max_side(img.convert("RGB"), thumb_max_side))
        except (FileNotFoundError, OSError):
            continue
    if not thumbs:
        return
    max_w = max(img.width for img in thumbs)
    max_h = max(img.height for img in thumbs)
    grid_cols = max(1, min(grid_cols, len(thumbs)))
    grid_rows = math.ceil(len(thumbs) / grid_cols)
    width = margin * 2 + grid_cols * max_w + (grid_cols - 1) * padding
    height = margin * 2 + grid_rows * max_h + (grid_rows - 1) * padding
    canvas = Image.new("RGB", (width, height), color=bg_color)
    for idx, thumb in enumerate(thumbs):
        row = idx // grid_cols
        col = idx % grid_cols
        x = margin + col * (max_w + padding)
        y = margin + row * (max_h + padding)
        canvas.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def export_frame_mosaic(
    frames: Sequence[Dict],
    frames_root: Optional[Path],
    expected_count: int,
    grid_cols: int,
    thumb_max_side: int,
    output_path: Path,
    sample_label: str = "",
) -> Optional[Path]:
    if frames_root is None or expected_count <= 0:
        return None
    resolved: List[Path] = []
    for frame in frames:
        path = resolve_frame_path(frame, frames_root)
        if path is not None:
            resolved.append(path)
    if not resolved:
        return None
    if len(resolved) < expected_count and sample_label:
        print(
            f"提示：{sample_label} 可用帧数 {len(resolved)} 低于目标 {expected_count}，将按现有帧数生成拼图。"
        )
    indices = select_frame_indices(len(resolved), expected_count)
    selected_paths = [resolved[i] for i in indices if i < len(resolved)]
    if not selected_paths:
        return None
    render_frame_grid(selected_paths, grid_cols, output_path, thumb_max_side)
    return output_path


def stratified_video_ids(video_ids: Sequence[str], desired: int, rng: random.Random):
    if not video_ids:
        return []
    sorted_ids = sorted(video_ids)
    if desired >= len(sorted_ids):
        shuffled = sorted_ids[:]
        rng.shuffle(shuffled)
        return shuffled
    bins = np.linspace(0, len(sorted_ids), desired + 1, dtype=int)
    chosen: List[str] = []
    used = set()
    for i in range(desired):
        start = bins[i]
        end = bins[i + 1]
        bucket = [vid for vid in sorted_ids[start:end] if vid not in used]
        if not bucket:
            bucket = [vid for vid in sorted_ids if vid not in used]
        if not bucket:
            break
        pick = rng.choice(bucket)
        chosen.append(pick)
        used.add(pick)
    if len(chosen) < desired:
        remaining = [vid for vid in sorted_ids if vid not in used]
        rng.shuffle(remaining)
        chosen.extend(remaining[: desired - len(chosen)])
    rng.shuffle(chosen)
    return chosen


def pick_segment(
    frames: Sequence[Dict],
    frames_per_sample: int,
    rng: random.Random,
    min_boxes: int,
    min_frames_required: int,
    max_attempts: int = 12,
) -> Optional[Dict]:
    if not frames:
        return None
    if len(frames) < max(1, min_frames_required):
        return None
    attempts = min(max_attempts, len(frames))
    for _ in range(attempts):
        if len(frames) <= frames_per_sample:
            start = 0
        else:
            start = rng.randint(0, max(len(frames) - frames_per_sample, 0))
        end = min(len(frames), start + frames_per_sample)
        segment = frames[start:end]
        if len(segment) < max(1, min_frames_required):
            continue
        total_boxes = sum(len(frame["boxes"]) for frame in segment)
        if total_boxes >= min_boxes:
            return {
                "frames": segment,
                "start_frame": segment[0]["frame_idx"],
                "end_frame": segment[-1]["frame_idx"],
                "total_boxes": total_boxes,
            }
    return None


def accumulate_heatmap(
    frames: Sequence[Dict], canvas_height: int, canvas_width: int
) -> np.ndarray:
    heatmap = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    for frame in frames:
        width = max(1, frame["width"])
        height = max(1, frame["height"])
        scale_x = canvas_width / width
        scale_y = canvas_height / height
        for box in frame["boxes"]:
            x1 = max(0, min(canvas_width - 1, int(math.floor(box[0] * scale_x))))
            y1 = max(0, min(canvas_height - 1, int(math.floor(box[1] * scale_y))))
            x2 = max(0, min(canvas_width, int(math.ceil(box[2] * scale_x))))
            y2 = max(0, min(canvas_height, int(math.ceil(box[3] * scale_y))))
            if x2 <= x1 or y2 <= y1:
                continue
            heatmap[y1:y2, x1:x2] += 1.0
    return heatmap


def render_heatmap(array: np.ndarray, title: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(array, cmap="magma")
    plt.title(title, fontsize=9)
    plt.axis("off")
    plt.tight_layout(pad=0.05)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def build_mosaic(
    image_paths: Sequence[Path],
    grid_cols: int,
    output_path: Path,
    padding: int = 12,
    margin: int = 24,
    bg_color=(15, 15, 15),
):
    if not image_paths:
        return
    images: List[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    grid_cols = max(1, min(grid_cols, len(images)))
    grid_rows = math.ceil(len(images) / grid_cols)
    mosaic_width = margin * 2 + grid_cols * max_w + (grid_cols - 1) * padding
    mosaic_height = margin * 2 + grid_rows * max_h + (grid_rows - 1) * padding
    canvas = Image.new("RGB", (mosaic_width, mosaic_height), color=bg_color)
    for idx, image in enumerate(images):
        if image.size != (max_w, max_h):
            image = image.resize((max_w, max_h), Image.BILINEAR)
        row = idx // grid_cols
        col = idx % grid_cols
        x = margin + col * (max_w + padding)
        y = margin + row * (max_h + padding)
        canvas.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    videos, max_height, max_width = load_dataset(args.labels)
    if not videos:
        raise RuntimeError(f"在 {args.labels} 中没有找到任何视频或标注。")
    canvas_height = args.canvas_height or max_height
    canvas_width = args.canvas_width or max_width
    video_ids = list(videos.keys())

    frames_root = args.frames_root
    if frames_root is not None and not frames_root.exists():
        print(f"警告：指定的 frames-root={frames_root} 不存在，将忽略帧导出。")
        frames_root = None
    if frames_root is None:
        candidates = [
            args.labels.parent / "frames",
            args.labels.parent.parent / "frames",
            args.labels.parent.parent / args.labels.parent.name / "frames",
        ]
        for candidate in candidates:
            try:
                if candidate.exists():
                    frames_root = candidate
                    print(f"提示：自动检测到帧目录 {frames_root}")
                    break
            except OSError:
                continue
    if frames_root is None:
        print("提示：未找到可用的帧目录，将仅输出热力图。")

    frames_per_image = args.frames_per_image
    if frames_per_image is None:
        frames_per_image = args.frames_per_sample
    if frames_per_image < 0:
        frames_per_image = 0
    if frames_per_image > args.frames_per_sample:
        print(
            f"提示：frames-per-image({frames_per_image}) 超过 frames-per-sample({args.frames_per_sample})，"
            "将按后者限制。"
        )
        frames_per_image = args.frames_per_sample
    min_frames_required = (
        max(1, min(args.frames_per_sample, frames_per_image))
        if frames_per_image > 0
        else 1
    )

    # 先按区间分层，保证抽样分布更分散。
    candidate_videos = stratified_video_ids(
        video_ids, min(args.samples * 3, len(video_ids)), rng
    )
    # 再追加所有视频作兜底，避免样本不足。
    remaining = [vid for vid in video_ids if vid not in candidate_videos]
    rng.shuffle(remaining)
    candidate_videos.extend(remaining)

    samples = []
    for video_id in candidate_videos:
        if len(samples) >= args.samples:
            break
        segment = pick_segment(
            videos[video_id],
            args.frames_per_sample,
            rng,
            args.min_boxes,
            min_frames_required=min_frames_required,
        )
        if segment is None:
            continue
        segment["video_id"] = video_id
        samples.append(segment)

    if len(samples) < args.samples:
        print(
            f"警告：仅成功抽取 {len(samples)} 个满足条件的样本，"
            f"请尝试降低 --min-boxes 或 --frames-per-sample。"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_output_dir = (
        args.frame_output_dir
        if args.frame_output_dir is not None
        else output_dir / "frame_mosaics"
    )
    heatmap_paths = []
    for idx, sample in enumerate(samples, start=1):
        heatmap = accumulate_heatmap(
            sample["frames"], canvas_height=canvas_height, canvas_width=canvas_width
        )
        title = (
            f"{sample['video_id']}"
            f" [{sample['start_frame']}-{sample['end_frame']}]"
            f" 共{sample['total_boxes']}框"
        )
        filename = (
            f"{idx:02d}_{sample['video_id']}_{sample['start_frame']}"
            f"-{sample['end_frame']}.png"
        )
        out_path = output_dir / filename
        render_heatmap(heatmap, title, out_path)
        mosaic_path = None
        if frames_root is not None and frames_per_image > 0:
            mosaic_name = f"{Path(filename).stem}_frames.png"
            mosaic_path = export_frame_mosaic(
                sample["frames"],
                frames_root,
                frames_per_image,
                args.frame_grid_cols,
                args.frame_thumb_max_side,
                frame_output_dir / mosaic_name,
                sample_label=title,
            )
        heatmap_paths.append(out_path)
        if mosaic_path is not None:
            print(f"[{idx:02d}] {title} -> 热力图: {out_path} | 帧拼图: {mosaic_path}")
        else:
            print(f"[{idx:02d}] {title} -> {out_path}")

    if not heatmap_paths:
        raise RuntimeError("未能生成任何热力图，请调整筛选参数后重试。")

    combined_output = (
        args.combined_output
        if args.combined_output is not None
        else output_dir / "heatmap_mosaic.png"
    )
    build_mosaic(heatmap_paths, args.grid_cols, combined_output)
    print(f"合成图已保存至 {combined_output.resolve()}")


if __name__ == "__main__":
    main()

