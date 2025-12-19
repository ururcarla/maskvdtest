"""
根据已转换的 Waymo COCO/VID 标注生成子集标注，避免拷贝图像。

默认策略：
- 选择 60 个 segment 的 FRONT 相机（每个 segment 只取 FRONT，共 60 条序列）。
- 另外选择 10 个 segment，取其余 4 路相机（FRONT_LEFT/FRONT_RIGHT/SIDE_LEFT/SIDE_RIGHT），共 40 条序列。
- 总计约 100 条序列。

可通过参数自定义数量或指定相机子集。输出新的 COCO/VID 标注文件，frames 仍指向原路径。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

CAM_SUFFIXES = [
    "-front",
    "-front_left",
    "-front_right",
    "-side_left",
    "-side_right",
]
FRONT_SUFFIX = "-front"
OTHER_SUFFIXES = [s for s in CAM_SUFFIXES if s != FRONT_SUFFIX]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coco-in",
        type=Path,
        default=None,
        help="输入 COCO 标注 (instances_x.json)，若缺失且提供 VID 将从 VID 构造 COCO。",
    )
    p.add_argument("--vid-in", type=Path, default=None, help="可选：输入 VID labels.json（可用于构造/子集）")
    p.add_argument("--out-root", type=Path, required=True, help="输出根目录，生成 annotations/ 下文件")
    p.add_argument("--split", choices=["train", "val"], default="train", help="split 名用于输出文件名")
    p.add_argument("--front-segments", type=int, default=60, help="取多少个 segment 的前向相机")
    p.add_argument("--other-segments", type=int, default=10, help="取多少个 segment 的其他四路相机")
    p.add_argument(
        "--cameras",
        nargs="+",
        default=CAM_SUFFIXES,
        help="允许的相机后缀（默认 5 路），如只训练前向可传 '-front'",
    )
    p.add_argument("--seed", type=int, default=42, help="随机种子（目前按排序截取，可忽略）")
    return p.parse_args()


def _parse_video_id(video_id: str) -> Tuple[str, str]:
    """从 video_id 获取 (segment, cam_suffix)。假设以相机后缀结尾。"""
    for suffix in CAM_SUFFIXES:
        if video_id.endswith(suffix):
            return video_id[: -len(suffix)], suffix
    return video_id, ""


def _filter_video_ids(video_ids: List[str], args: argparse.Namespace) -> Set[str]:
    allowed = set()
    front_segments: List[str] = []
    other_segments: List[str] = []

    # 先按 camera 过滤
    candidates = []
    for vid in video_ids:
        seg, cam = _parse_video_id(vid)
        if args.cameras and cam not in args.cameras:
            continue
        candidates.append((seg, cam, vid))

    # FRONT
    for seg, cam, vid in candidates:
        if cam == FRONT_SUFFIX:
            if seg in front_segments:
                continue
            if len(front_segments) >= args.front_segments:
                continue
            front_segments.append(seg)
            allowed.add(vid)

    # OTHER cams：每个 segment 取所有非 front 相机
    for seg, cam, vid in candidates:
        if cam in OTHER_SUFFIXES and cam in args.cameras:
            if len(other_segments) >= args.other_segments and seg not in other_segments:
                continue
            if seg not in other_segments:
                other_segments.append(seg)
            allowed.add(vid)

    return allowed


def _subset_coco(coco_in: Path, out_file: Path, keep_videos: Set[str]):
    data = json.loads(coco_in.read_text())
    seq_dirs = data.get("seq_dirs") or []
    # 选中序列并重排 sid
    new_seq_dirs = [vid for vid in seq_dirs if vid in keep_videos]
    seq_id_map = {vid: idx for idx, vid in enumerate(new_seq_dirs)}

    images = []
    annotations = []
    for img in data["images"]:
        vid = seq_dirs[img["sid"]]
        if vid not in keep_videos:
            continue
        new_sid = seq_id_map[vid]
        new_img = dict(img)
        new_img["sid"] = new_sid
        images.append(new_img)

    keep_image_ids = {img["id"] for img in images}
    ann_id = 1
    for ann in data["annotations"]:
        if ann["image_id"] not in keep_image_ids:
            continue
        new_ann = dict(ann)
        new_ann["id"] = ann_id
        ann_id += 1
        annotations.append(new_ann)

    data_out = {
        "seq_dirs": new_seq_dirs,
        "images": images,
        "annotations": annotations,
        "categories": data.get("categories", []),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(data_out))
    print(f"[subset] COCO saved to {out_file} "
          f"(seq={len(new_seq_dirs)}, images={len(images)}, anns={len(annotations)})")


def _subset_vid(vid_in: Path, out_file: Path, keep_videos: Set[str]):
    data = json.loads(vid_in.read_text())
    images = []
    annotations = []
    ann_id = 1
    for img in data["images"]:
        # file_name 形如 <video_id>/<video_id>_000123.jpg
        video_id = Path(img["file_name"]).parts[0]
        if video_id not in keep_videos:
            continue
        images.append(img)
    keep_ids = {img["id"] for img in images}
    for ann in data["annotations"]:
        if ann["image_id"] not in keep_ids:
            continue
        new_ann = dict(ann)
        new_ann["id"] = ann_id
        ann_id += 1
        annotations.append(new_ann)
    data_out = {
        "images": images,
        "annotations": annotations,
        "categories": data.get("categories", []),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(data_out))
    print(f"[subset] VID saved to {out_file} "
          f"(images={len(images)}, anns={len(annotations)})")


def _build_coco_from_vid(vid_data: Dict) -> Dict:
    """
    将 ImageNet-VID 风格的 labels.json 转为 COCO 视频格式，便于子集和 ArgoverseVID 读取。
    假设 file_name 形如 <video_id>/<video_id>_000123.jpg。
    """
    # 聚合按 video_id
    frames_by_video: Dict[str, List[Dict]] = {}
    for img in vid_data.get("images", []):
        file_name = img["file_name"]
        parts = Path(file_name).parts
        if len(parts) < 2:
            continue
        video_id = parts[0]
        frames_by_video.setdefault(video_id, []).append(img)

    # 便于查找 annotations
    anns_by_image = {}
    for ann in vid_data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    seq_dirs = sorted(frames_by_video.keys())
    seq_id_map = {vid: sid for sid, vid in enumerate(seq_dirs)}

    images = []
    annotations = []
    ann_id = 1
    image_id = 1
    for vid in seq_dirs:
        frames = sorted(frames_by_video[vid], key=lambda x: x["file_name"])
        for fid, img in enumerate(frames):
            name = Path(img["file_name"]).name
            images.append(
                {
                    "id": image_id,
                    "sid": seq_id_map[vid],
                    "fid": fid,
                    "name": name,
                    "height": img["height"],
                    "width": img["width"],
                }
            )
            for ann in anns_by_image.get(img["id"], []):
                x, y, w, h = ann["bbox"]
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": ann["category_id"],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": ann.get("iscrowd", 0),
                    }
                )
                ann_id += 1
            image_id += 1

    return {
        "seq_dirs": seq_dirs,
        "images": images,
        "annotations": annotations,
        "categories": vid_data.get("categories", []),
    }


def main():
    args = parse_args()
    coco_path = args.coco_in
    vid_path = args.vid_in
    if coco_path is None and vid_path is None:
        raise FileNotFoundError("至少需要提供 --coco-in 或 --vid-in 之一。")

    # 如果没有 COCO 输入但有 VID，则先用 VID 构造 COCO 数据体
    coco_data = None
    if coco_path and coco_path.is_file():
        coco_data = json.loads(coco_path.read_text())
    elif vid_path and vid_path.is_file():
        coco_data = _build_coco_from_vid(json.loads(vid_path.read_text()))
    else:
        raise FileNotFoundError("找不到 COCO 或 VID 输入文件。")

    keep_videos = _filter_video_ids(coco_data.get("seq_dirs", []), args)
    if not keep_videos:
        raise RuntimeError("未选出任何序列，请检查相机后缀或数量参数。")

    ann_dir = args.out_root / args.split / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    coco_out = ann_dir / f"instances_{args.split}.json"
    # 先把 coco_data 写到临时变量再子集，避免需要文件存在
    tmp_coco_path = coco_out.with_suffix(".tmp.json")
    tmp_coco_path.write_text(json.dumps(coco_data))
    _subset_coco(tmp_coco_path, coco_out, keep_videos)
    tmp_coco_path.unlink(missing_ok=True)

    if vid_path:
        vid_out = args.out_root / args.split / "labels.json"
        _subset_vid(vid_path, vid_out, keep_videos)


if __name__ == "__main__":
    main()
