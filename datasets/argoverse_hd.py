import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from datasets.vid import VIDItem
from utils.misc import seeded_shuffle


SPLITS = {"train", "val", "test"}


class ArgoverseHD(Dataset):
    """
    Dataset wrapper for Argoverse-HD that mimics the datasets.VID output.

    Each __getitem__ returns a VIDItem representing a single driving sequence,
    so it can be dropped in wherever the original VID loader was used.
    """

    def __init__(
        self,
        location: Path,
        split: str = "train",
        annotations_path: Optional[Path] = None,
        verify_frames: bool = True,
        shuffle: bool = True,
        shuffle_seed: int = 42,
        frame_transform=None,
        annotation_transform=None,
        combined_transform=None,
    ) -> None:
        assert split in SPLITS, f"split must be one of {SPLITS}"
        self.location = Path(location)
        self.split = split
        self.frame_transform = frame_transform
        self.annotation_transform = annotation_transform
        self.combined_transform = combined_transform
        self.verify_frames = verify_frames

        self.images_root = self.location / "Argoverse-1.1" / "tracking"
        self.split_root = self.images_root / split
        self.annotations_path = (
            Path(annotations_path)
            if annotations_path is not None
            else self._default_annotations_path(split)
        )

        self.video_info = self._build_video_index()
        if shuffle:
            seeded_shuffle(self.video_info, shuffle_seed)

    def __len__(self) -> int:
        return len(self.video_info)

    def __getitem__(self, index: int) -> VIDItem:
        video_info = self.video_info[index]
        frame_paths = [frame["path"] for frame in video_info["frames"]]
        annotations = [frame["annotations"] for frame in video_info["frames"]]
        vid_item = VIDItem(
            frame_paths,
            annotations,
            self.frame_transform,
            self.annotation_transform,
            self.combined_transform,
        )
        return vid_item

    def _default_annotations_path(self, split: str) -> Path:
        name = "test-meta.json" if split == "test" else f"{split}.json"
        return self.location / "Argoverse-HD" / "annotations" / name

    def _build_category_mapping(self, categories: Sequence[Dict]) -> Dict[int, int]:
        """
        Argoverse-HD categories are not guaranteed to be zero-based.
        Map raw category ids → contiguous [0, N) indices to match ViTDet.
        """
        if not categories:
            return {}
        mapping = {}
        for new_id, cat in enumerate(categories):
            mapping[cat["id"]] = new_id
        return mapping

    def _build_video_index(self) -> List[Dict]:
        with self.annotations_path.open("r") as fp:
            json_data = json.load(fp)

        category_mapping = self._build_category_mapping(json_data.get("categories", []))

        frame_dict: Dict[int, Dict] = {}
        for image_meta in json_data.get("images", []):
            frame = self._create_frame_entry(image_meta)
            frame_dict[image_meta["id"]] = frame

        for ann in json_data.get("annotations", []):
            entry = frame_dict.get(ann["image_id"])
            if entry is None:
                continue
            boxes = entry["annotations"]["boxes"]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels = entry["annotations"]["labels"]
            if category_mapping:
                labels.append(category_mapping[ann["category_id"]])
            else:
                labels.append(ann["category_id"] - 1)

        for frame in frame_dict.values():
            annotation = frame["annotations"]
            boxes = torch.tensor(annotation["boxes"], dtype=torch.float32)
            labels = torch.tensor(annotation["labels"], dtype=torch.int64)
            annotation["boxes"] = boxes
            annotation["labels"] = labels

        sequence_dict: Dict[str, List[Dict]] = defaultdict(list)
        for frame in frame_dict.values():
            sequence_dict[frame["sequence_id"]].append(frame)

        video_info = []
        for seq_id, frames in sequence_dict.items():
            frames.sort(key=self._frame_sort_key)
            video_info.append({"video_id": seq_id, "frames": frames})

        video_info.sort(key=lambda item: item["video_id"])
        return video_info

    def _create_frame_entry(self, image_meta: Dict) -> Dict:
        seq_id = (
            image_meta.get("sequence_id")
            or image_meta.get("sequence")
            or image_meta.get("video_id")
        )
        if seq_id is None:
            seq_id = Path(image_meta["file_name"]).parent.name

        timestamp = image_meta.get("timestamp")
        frame_id = image_meta.get("frame_id")
        file_name = image_meta["file_name"]
        frame_path = self._resolve_frame_path(seq_id, file_name)

        annotation = {"boxes": [], "labels": []}
        if frame_id is not None:
            annotation["frame_id"] = torch.tensor([frame_id])
        if timestamp is not None:
            annotation["timestamp"] = torch.tensor([timestamp])

        return {
            "sequence_id": seq_id,
            "path": str(frame_path),
            "timestamp": timestamp,
            "frame_id": frame_id,
            "filename": Path(file_name).name,
            "annotations": annotation,
        }

    def _resolve_frame_path(self, sequence_id: str, file_name: str) -> Path:
        """
        Resolve the actual image path. Argoverse-HD may store frames either directly
        under the sequence directory or inside a ring_front_center/ subdirectory.
        """
        file_path = Path(file_name)
        candidates = []

        if file_path.is_absolute():
            candidates.append(file_path)
        else:
            candidates.extend(
                [
                    self.images_root / file_path,
                    self.split_root / file_path,
                    self.split_root / sequence_id / file_path.name,
                    self.split_root / sequence_id / "ring_front_center" / file_path.name,
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        if self.verify_frames:
            candidate_list = "\n".join(str(c) for c in candidates)
            raise FileNotFoundError(
                f"Could not locate frame '{file_name}' for sequence '{sequence_id}'. "
                f"Tried:\n{candidate_list}"
            )

        return candidates[0]

    @staticmethod
    def _frame_sort_key(frame_entry: Dict):
        # Prefer explicit frame_id, then timestamp, finally filename.
        frame_id = frame_entry.get("frame_id")
        timestamp = frame_entry.get("timestamp")
        filename = frame_entry["filename"]
        return (
            0 if frame_id is not None else 1,
            frame_id if frame_id is not None else 0,
            0 if timestamp is not None else 1,
            timestamp if timestamp is not None else 0,
            filename,
        )

