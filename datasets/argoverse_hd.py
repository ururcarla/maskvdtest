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
        self.sequence_dirs_by_sid: Dict[int, Path] = {}
        self.sequence_name_by_sid: Dict[int, str] = {}
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

        (
            self.sequence_dirs_by_sid,
            self.sequence_name_by_sid,
        ) = self._prepare_sequence_metadata(json_data)

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
        sid = image_meta.get("sid")
        seq_id = (
            image_meta.get("sequence_id")
            or image_meta.get("sequence")
            or image_meta.get("video_id")
        )
        if seq_id is None and sid is not None:
            seq_id = self.sequence_name_by_sid.get(int(sid))
        file_name = image_meta.get("file_name") or image_meta.get("name")
        if seq_id is None:
            if file_name and "/" in file_name:
                seq_id = Path(file_name).parent.name
            elif sid is not None:
                seq_id = f"sid_{sid}"
            else:
                seq_id = str(image_meta.get("id"))
        seq_id = str(seq_id)

        timestamp = image_meta.get("timestamp")
        if timestamp is None:
            timestamp = image_meta.get("time")
        frame_id = image_meta.get("frame_id")
        if frame_id is None:
            frame_id = image_meta.get("fid")
        if file_name is None:
            raise KeyError("file_name")

        seq_dir = None
        if sid is not None:
            seq_dir = self.sequence_dirs_by_sid.get(int(sid))

        frame_path = self._resolve_frame_path(seq_id, file_name, seq_dir)

        annotation = {"boxes": [], "labels": []}
        if frame_id is not None:
            frame_id_int = int(frame_id)
            annotation["frame_id"] = torch.tensor([frame_id_int])
        if timestamp is not None:
            timestamp_val = int(timestamp)
            annotation["timestamp"] = torch.tensor([timestamp_val])

        return {
            "sequence_id": seq_id,
            "path": str(frame_path),
            "timestamp": timestamp,
            "frame_id": frame_id,
            "filename": Path(file_name).name,
            "annotations": annotation,
        }

    def _resolve_frame_path(
        self, sequence_id: str, file_name: str, sequence_dir: Optional[Path] = None
    ) -> Path:
        """
        Resolve the actual image path. Argoverse-HD may store frames either directly
        under the sequence directory or inside a ring_front_center/ subdirectory.
        """
        file_path = Path(file_name)
        candidates = []

        if file_path.is_absolute():
            candidates.append(file_path)
        else:
            if sequence_dir is not None:
                candidates.append(sequence_dir / file_path.name)
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

    def _prepare_sequence_metadata(
        self, json_data: Dict
    ) -> (Dict[int, Path], Dict[int, str]):
        dir_map: Dict[int, Path] = {}
        name_map: Dict[int, str] = {}

        sid_values = [
            img.get("sid")
            for img in json_data.get("images", [])
            if isinstance(img.get("sid"), (int, float))
        ]
        sid_offset = int(min(sid_values)) if sid_values else 0

        seq_dirs = json_data.get("seq_dirs")
        if isinstance(seq_dirs, (list, tuple)):
            for idx, seq_dir in enumerate(seq_dirs):
                sid = sid_offset + idx
                resolved = self._normalize_seq_dir(seq_dir)
                if resolved is not None:
                    dir_map[sid] = resolved
                if seq_dir:
                    name_map.setdefault(sid, Path(seq_dir).stem)

        sequences = json_data.get("sequences")
        if isinstance(sequences, list):
            for seq in sequences:
                sid = seq.get("sid")
                if sid is None:
                    sid = seq.get("id")
                if sid is None:
                    continue
                sid = int(sid)
                name = seq.get("name") or seq.get("sequence_name") or seq.get("dir")
                if name:
                    name_map[sid] = name
                seq_dir_value = seq.get("dir") or seq.get("path") or seq.get("seq_dir")
                resolved = self._normalize_seq_dir(seq_dir_value)
                if resolved is not None:
                    dir_map[sid] = resolved

        return dir_map, name_map

    def _normalize_seq_dir(self, seq_dir: Optional[str]) -> Optional[Path]:
        if not seq_dir:
            return None
        path = Path(seq_dir)
        if path.is_absolute():
            return path

        candidate_roots = [
            self.location,
            self.images_root,
            self.split_root,
        ]
        for root in candidate_roots:
            candidate = root / path
            if candidate.exists():
                return candidate
        return (self.location / path).resolve()

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

