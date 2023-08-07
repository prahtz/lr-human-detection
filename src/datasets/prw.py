import random
from pathlib import Path

import h5py
import numpy as np
import torch
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
        mask = np.ones(im_shape, dtype=np.bool_)
        for bbox in bboxes:
            x, y, w, h = bbox
            mask[y : y + h, x : x + w] = False
        w, h = anchor_wh
        anchor_dim = max(w, h)
        mask[-anchor_dim:, :] = False
        mask[:, -anchor_dim:] = False
        for bbox in bboxes:
            x, y, w, h = bbox
            mask[max(0, y - anchor_dim + 1) : y + h, max(0, x - anchor_dim + 1) : x + w] = False
        sample_space = np.argwhere(mask)
        idx = random.randint(0, len(sample_space) - 1)
        return sample_space[idx]

    def __getitem__(self, idx: int):
        img_name = self.idx_to_img_name[idx]
        im_h, im_w = self.img_group[img_name].shape[:2]
        bboxes = self.img_group[img_name].attrs["bboxes"]
        anchor_id = np.random.choice(len(self.anchors_weights), p=self.anchors_weights)
        anchor_wh = self.anchors_dict[str(im_w)][anchor_id]
        anchor_xy = self.sample_anchor_position(anchor_wh, bboxes, (im_w, im_h))
        anchor_size = max(anchor_wh)
        image = self.img_group[img_name][anchor_xy[1] : anchor_xy[1] + anchor_size, anchor_xy[0] : anchor_xy[0] + anchor_size]
        return torch.from_numpy(image).permute(2, 0, 1), torch.tensor(0).long()

    def __del__(self):
        self.dataset_file.close()


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
