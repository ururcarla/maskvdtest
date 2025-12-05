from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from datasets.vid import VIDItem
from utils.misc import seeded_shuffle


@dataclass
class _FrameRecord:
    path: str
    annotations: Dict[str, torch.Tensor]


class ArgoverseVID(Dataset):
    """Video-style dataset wrapper around Argoverse-HD annotations."""

    def __init__(
        self,
        img_root: Path | str,
        ann_file: Path | str,
        *,
        shuffle: bool = True,
        shuffle_seed: int = 42,
        frame_transform=None,
        annotation_transform=None,
        combined_transform=None,
        min_box_size: int = 2,
    ):
        self.img_root = Path(img_root)
        self.ann_file = Path(ann_file)
        self.frame_transform = frame_transform
        self.annotation_transform = annotation_transform
        self.combined_transform = combined_transform
        self.min_box_size = min_box_size

        self.coco = COCO(str(self.ann_file))
        self.class_ids = sorted(self.coco.getCatIds())
        self.class_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.class_ids)}

        self.video_info = self._build_video_info()
        if shuffle:
            seeded_shuffle(self.video_info, shuffle_seed)

    def __len__(self) -> int:
        return len(self.video_info)

    def __getitem__(self, index: int):
        video = self.video_info[index]
        frame_paths = [frame.path for frame in video["frames"]]
        annotations = [frame.annotations for frame in video["frames"]]
        return VIDItem(
            frame_paths,
            annotations,
            self.frame_transform,
            self.annotation_transform,
            self.combined_transform,
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _build_video_info(self) -> List[Dict]:
        dataset = self.coco.dataset
        seq_dirs = dataset.get("seq_dirs")
        if seq_dirs is None:
            raise ValueError("Argoverse annotations must provide 'seq_dirs'.")
        seq_dirs = [Path(d) for d in seq_dirs]

        img_ids = self.coco.getImgIds()
        if len(img_ids) == 0:
            return []

        seq_lens: List[int] = []
        first_img_id: Optional[int] = None
        first_seq_id: Optional[int] = None
        for i, image_id in enumerate(img_ids):
            img = self.coco.loadImgs([image_id])[0]
            if i == 0:
                first_img_id = image_id
                first_seq_id = img["sid"]
                seq_lens.append(1)
                continue

            assert first_img_id is not None
            assert first_seq_id is not None
            assert i + first_img_id == image_id, "Image ids must be contiguous."

            expected_sid = first_seq_id + len(seq_lens) - 1
            if img["sid"] == expected_sid:
                assert (
                    img["fid"] == seq_lens[-1]
                ), "Frame ids must increment within a sequence."
                seq_lens[-1] += 1
            else:
                assert (
                    img["sid"] == first_seq_id + len(seq_lens)
                ), "Sequence ids must be contiguous."
                assert img["fid"] == 0, "Sequences must start at frame id 0."
                seq_lens.append(1)

        assert first_img_id is not None
        assert first_seq_id is not None

        if len(seq_dirs) != len(seq_lens):
            raise ValueError(
                f"'seq_dirs' ({len(seq_dirs)}) and detected sequences ({len(seq_lens)}) mismatch."
            )

        seq_start_img_ids = list(self._exclusive_cumsum(first_img_id, seq_lens))

        videos: List[Dict] = []
        for seq_idx, (seq_len, seq_dir, seq_start_img_id) in enumerate(
            zip(seq_lens, seq_dirs, seq_start_img_ids)
        ):
            frames: List[_FrameRecord] = []
            for frame_id in range(seq_len):
                image_id = seq_start_img_id + frame_id
                img_ann = self.coco.loadImgs([image_id])[0]
                frame = self._build_frame_record(seq_dir, img_ann)
                frames.append(frame)

            videos.append(
                {
                    "seq_id": (first_seq_id or 0) + seq_idx,
                    "frames": frames,
                }
            )
        return videos

    def _build_frame_record(self, seq_dir: Path, img_ann: Dict) -> _FrameRecord:
        image_path = str(self.img_root / seq_dir / img_ann["name"])
        annotations = self._load_annotations(img_ann)
        return _FrameRecord(path=image_path, annotations=annotations)

    def _load_annotations(self, img_ann: Dict) -> Dict[str, torch.Tensor]:
        img_height = img_ann["height"]
        img_width = img_ann["width"]
        image_id = img_ann["id"]

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        ann_list = self.coco.loadAnns(ann_ids)

        if len(ann_list) == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "track_ids": torch.zeros((0,), dtype=torch.int64),
            }

        boxes = torch.tensor([ann["bbox"] for ann in ann_list], dtype=torch.float32)
        labels = torch.tensor(
            [self.class_id_to_idx[ann["category_id"]] for ann in ann_list],
            dtype=torch.int64,
        )
        track_ids = torch.tensor(
            [ann.get("track", -1) for ann in ann_list], dtype=torch.int64
        )

        # Convert xywh to xyxy and clip to the image boundary.
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp_(min=0, max=img_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp_(min=0, max=img_height)

        wh = boxes[:, 2:] - boxes[:, :2]
        valid = torch.min(wh, dim=1)[0] >= float(self.min_box_size)
        if not torch.all(valid):
            boxes = boxes[valid]
            labels = labels[valid]
            track_ids = track_ids[valid]

        return {"boxes": boxes, "labels": labels, "track_ids": track_ids}

    @staticmethod
    def _exclusive_cumsum(start: int, seq_lens: List[int]):
        current = start
        for length in seq_lens:
            yield current
            current += length

