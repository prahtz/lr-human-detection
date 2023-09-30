import random
from pathlib import Path

import h5py
import numpy as np
import scipy
import torch
from PIL import Image
from torch.utils.data import Dataset


class Positives(Dataset):
    def __init__(self, root_path: str, split: str = "train") -> None:
        self.root_path = Path(root_path)
        self.split = split
        self.dataset_file = h5py.File(self.root_path / "positives.h5", "r")

    def __len__(self) -> int:
        return len(self.dataset_file[self.split])

    def __getitem__(self, idx: int):
        split_group = self.dataset_file[self.split]
        image = split_group[f"img_{idx}"][:]
        return torch.from_numpy(image).permute(2, 0, 1), torch.scalar_tensor(1).long()

    def __del__(self):
        self.dataset_file.close()


class Negatives(Dataset):
    def __init__(self, root_path: str, split: str = "train") -> None:
        self.root_path = Path(root_path)
        self.split = split
        self.dataset_file = h5py.File(self.root_path / "negatives.h5", "r")

        self.anchors_group = self.dataset_file["anchors"]
        self.split_group = self.dataset_file[self.split]
        self.img_group = self.split_group["images"]

        img_names = [k for k in self.img_group.keys()]
        self.img_to_num_samples = {k: len(self.img_group[k].attrs["bboxes"]) for k in img_names}
        self.idx_to_img_name = [k for k, v in self.img_to_num_samples.items() for _ in range(v)]
        self.total_len = sum([v for v in self.img_to_num_samples.values()])
        self.anchors_dict = {k: v for k, v in self.anchors_group.attrs.items()}
        self.anchors_weights = self.anchors_group.attrs["anchors_weights"]

    def __len__(self) -> int:
        return self.total_len

    def sample_anchor_position(self, anchor_wh, bboxes, im_shape):
        width, height = im_shape
        w, h = anchor_wh
        anchor_dim = max(w, h)
        width = width - anchor_dim
        height = height - anchor_dim
        # Add 1D index intervals where anchor position can not be sampled
        ranges = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x, y = max(0, x), max(0, y)

            for k in range(y, y + h):
                i = k * width + min(x, width)
                j = k * width + min(x + w, width)
                ranges.append([i, j])

            for k in range(max(0, y - anchor_dim + 1), y + h):
                i = k * width + min(max(0, x - anchor_dim + 1), width)
                j = k * width + min(x + w, width)
                ranges.append([i, j])

        # Merge overlapped and duplicated intervals
        ranges = sorted(ranges)
        idx = 0
        for i in range(1, len(ranges)):
            if ranges[idx][1] >= ranges[i][0]:
                ranges[idx][1] = max(ranges[idx][1], ranges[i][1])
            else:
                idx += 1
                ranges[idx] = ranges[i]
        ranges = ranges[: idx + 1]

        # Compute intervals where anchors can be sampled
        intervals, probs = [], []
        last, total_length = 0, 0
        for r in ranges:
            a, b = r
            a = min(a, height * width)
            intervals.append([last, a - 1])
            interval_length = a - last
            probs.append(interval_length)
            total_length += interval_length
            last = b
            if last >= height * width:
                break
        if last < height * width:
            intervals.append([last, height * width - 1])
            probs.append(height * width - last)
            total_length += height * width - last
        if total_length == 0:
            return (0, 0)
        probs = [p / total_length for p in probs]

        # Sample interval according to its frequency probability and then sample the index from it uniformly
        idx = np.random.choice(len(intervals), p=probs)
        idx = random.randint(intervals[idx][0], intervals[idx][1])
        return (idx % width, idx // width)

    def __getitem__(self, idx: int):
        img_name = self.idx_to_img_name[idx]
        im_h, im_w = self.img_group[img_name].shape[:2]
        bboxes = self.img_group[img_name].attrs["bboxes"]
        anchor_id = np.random.choice(len(self.anchors_weights), p=self.anchors_weights)
        anchor_wh = self.anchors_dict[str(im_w)][anchor_id]
        x, y = self.sample_anchor_position(anchor_wh, bboxes, (im_w, im_h))
        anchor_size = max(anchor_wh)
        image = self.img_group[img_name][y : y + anchor_size, x : x + anchor_size]
        return torch.from_numpy(image).permute(2, 0, 1), torch.tensor(0).long()

    def __del__(self):
        self.dataset_file.close()


class PRWRawDataset(Dataset):
    __ALLOWED_SPLITS__ = ["train", "valid", "test"]

    def __init__(self, root_path: str, split: str = "train", shuffle=True) -> None:
        assert split in self.__ALLOWED_SPLITS__

        self.root_path = Path(root_path)
        self.split = split
        self.annotations_dir_path = self.root_path / "annotations/"

        split_filename = "test" if split == "test" else "train"
        split_idxs = scipy.io.loadmat(self.root_path / f"frame_{split_filename}.mat")
        frame_names = sorted([img_idx[0][0] + ".jpg" for img_idx in split_idxs[f"img_index_{split_filename}"]])
        annotations_paths = [self.annotations_dir_path / (frame_name + ".mat") for frame_name in frame_names]

        self.annotations = []
        for frame_name, annotation_path in zip(frame_names, annotations_paths):
            annotation = scipy.io.loadmat(annotation_path)
            if "box_new" in annotation:
                bboxes = annotation["box_new"]
            elif "anno_file" in annotation:
                bboxes = annotation["anno_file"]
            else:
                bboxes = annotation["anno_previous"]
            bboxes = [list(map(round, bbox[1:])) for bbox in bboxes]
            camera = int(frame_name[1])
            sequence = int(frame_name[3])
            self.annotations.append(
                {"img_path": self.root_path / ("frames/" + frame_name), "bboxes": bboxes, "video_id": (camera, sequence)}
            )

        if split != "test":
            train_val_split_idx = int(len(self.annotations) * 0.8)
            if shuffle:
                random.seed(42)
                random.shuffle(self.annotations)
            self.annotations = self.annotations[:train_val_split_idx] if split == "train" else self.annotations[train_val_split_idx:]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        img_path, bboxes = item["img_path"], item["bboxes"]
        img = np.array(Image.open(img_path))
        return img, bboxes, item["video_id"]


class PRWClassification(Dataset):
    def __init__(self, root_path: str, split: str = "train") -> None:
        self.positives = Positives(root_path=root_path, split=split)
        self.negatives = Negatives(root_path=root_path, split=split)
        self.total_size = len(self.positives) + len(self.negatives)

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int):
        if idx < len(self.positives):
            return self.positives.__getitem__(idx)
        return self.negatives.__getitem__(idx - len(self.positives))
