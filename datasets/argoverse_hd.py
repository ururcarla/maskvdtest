import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from datasets.vid import VID, VIDItem
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
        prepared_root: Optional[Path] = None,
        auto_prepare: bool = True,
        copy_images: bool = False,
    ) -> None:
        assert split in SPLITS, f"split must be one of {SPLITS}"
        self.location = Path(location)
        self.split = split
        self.frame_transform = frame_transform
        self.annotation_transform = annotation_transform
        self.combined_transform = combined_transform
        self.verify_frames = verify_frames
        self.auto_prepare = auto_prepare
        self.copy_images = copy_images

        self.images_root = self.location / "Argoverse-1.1" / "tracking"
        self.split_root = self.images_root / split
        self.sequence_dirs_by_sid: Dict[int, Path] = {}
        self.sequence_name_by_sid: Dict[int, str] = {}
        self.annotations_path = (
            Path(annotations_path)
            if annotations_path is not None
            else self._default_annotations_path(split)
        )

        self.prepared_root = (
            Path(prepared_root)
            if prepared_root is not None
            else self.location / "prepared_vid"
        )
        self._prepared_dataset: Optional[VID] = self._try_load_prepared_dataset(
            split, shuffle, shuffle_seed
        )
        if self._prepared_dataset is not None:
            self.video_info = self._prepared_dataset.video_info
            return

        json_data = self._load_annotations()
        self.video_info = self._build_video_index(json_data)
        if shuffle:
            seeded_shuffle(self.video_info, shuffle_seed)

        if self.auto_prepare:
            self._prepare_vid_layout(json_data)
            prepared = self._try_load_prepared_dataset(split, shuffle, shuffle_seed)
            if prepared is not None:
                self._prepared_dataset = prepared
                self.video_info = prepared.video_info

    def __len__(self) -> int:
        if self._prepared_dataset is not None:
            return len(self._prepared_dataset)
        return len(self.video_info)

    def __getitem__(self, index: int) -> VIDItem:
        if self._prepared_dataset is not None:
            return self._prepared_dataset[index]
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

    def _prepared_split_name(self) -> str:
        if self.split in {"train", "val"}:
            return f"vid_{self.split}"
        return self.split

    def _prepared_split_root(self) -> Path:
        return self.prepared_root / self._prepared_split_name()

    def _has_prepared_split(self) -> bool:
        split_root = self._prepared_split_root()
        return (split_root / "labels.json").is_file()

    def _try_load_prepared_dataset(
        self, split: str, shuffle: bool, shuffle_seed: int
    ) -> Optional[VID]:
        if not self._has_prepared_split():
            return None
        prepared_split = self._prepared_split_name()
        return VID(
            self.prepared_root,
            split=prepared_split,
            tar_path=None,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            frame_transform=self.frame_transform,
            annotation_transform=self.annotation_transform,
            combined_transform=self.combined_transform,
        )

    def _load_annotations(self) -> Dict:
        with self.annotations_path.open("r") as fp:
            return json.load(fp)

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

    def _build_video_index(self, json_data: Dict) -> List[Dict]:
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

        height = image_meta.get("height")
        width = image_meta.get("width")

        seq_dir = None
        if sid is not None:
            seq_dir = self.sequence_dirs_by_sid.get(int(sid))

        frame_path = self._resolve_frame_path(seq_id, file_name, seq_dir)

        annotation = {"boxes": [], "labels": [], "height": height, "width": width}
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

    def _prepare_vid_layout(self, json_data: Dict) -> None:
        split_root = self._prepared_split_root()
        labels_path = split_root / "labels.json"
        if labels_path.exists():
            return
        frames_root = split_root / "frames"
        frames_root.mkdir(parents=True, exist_ok=True)

        categories = json_data.get("categories", [])
        images = []
        annotations = []
        videos = []
        image_id = 0
        annotation_id = 0

        for video in tqdm_list(self.video_info, desc=f"Prepare {self.split}"):
            video_id = video["video_id"]
            videos.append({"id": video_id, "file_name": video_id})
            seq_dir = frames_root / video_id
            seq_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx, frame in enumerate(video["frames"]):
                src_path = Path(frame["path"])
                suffix = src_path.suffix or ".jpg"
                dest_name = f"{frame_idx:06d}{suffix}"
                dest_path = seq_dir / dest_name
                if not src_path.exists():
                    if self.verify_frames:
                        raise FileNotFoundError(f"Missing frame {src_path}")
                    else:
                        continue
                replicate_file(src_path, dest_path, self.copy_images)

                metadata = frame["annotations"]
                height = metadata.get("height")
                width = metadata.get("width")
                if (height is None or width is None) and dest_path.exists():
                    try:
                        with Image.open(dest_path) as img:
                            width, height = img.size
                    except Exception:
                        width = width or 0
                        height = height or 0

                images.append(
                    {
                        "id": image_id,
                        "file_name": f"{video_id}/{dest_name}",
                        "video_id": video_id,
                        "frame_id": frame_idx,
                        "height": height,
                        "width": width,
                    }
                )

                boxes = metadata.get("boxes")
                labels = metadata.get("labels")
                if boxes is not None and labels is not None and boxes.numel() > 0:
                    boxes_list = _tensor_to_list(boxes)
                    labels_list = _tensor_to_list(labels)
                    for box, label in zip(boxes_list, labels_list):
                        x1, y1, x2, y2 = map(float, box)
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        annotations.append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": int(label) + 1,
                                "bbox": [x1, y1, w, h],
                                "area": w * h,
                                "iscrowd": 0,
                            }
                        )
                        annotation_id += 1
                image_id += 1

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

        # Mark the prepared directory as "unpacked" so downstream VID loaders
        # know they don't need the original tar archives.
        unpacked_flag = self.prepared_root / "unpacked"
        unpacked_flag.parent.mkdir(parents=True, exist_ok=True)
        unpacked_flag.touch()

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
            for idx, seq in enumerate(sequences):
                if isinstance(seq, str):
                    resolved = self._normalize_seq_dir(seq)
                    if resolved is not None:
                        sid = sid_offset + idx
                        dir_map[sid] = resolved
                        name_map[sid] = Path(seq).stem
                    continue
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


def replicate_file(src: Path, dst: Path, force_copy: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if force_copy:
            shutil.copy2(src, dst)
        else:
            os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _tensor_to_list(tensor) -> List:
    if tensor is None:
        return []
    if hasattr(tensor, "tolist"):
        return tensor.tolist()
    return list(tensor)


def tqdm_list(data: List[Dict], desc: str):
    try:
        return tqdm(data, desc=desc, unit="video")
    except Exception:
        return data

