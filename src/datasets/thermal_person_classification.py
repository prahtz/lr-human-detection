import random

import h5py
import torch
from torch.utils.data import Dataset


class Positives(Dataset):
    def __init__(self, root_path: str, split: str = "train") -> None:
        self.root_path = root_path
        self.split = split
        self.dataset_file = h5py.File(self.root_path, "r")

    def __len__(self) -> int:
        return len(self.dataset_file[self.split])

    def __getitem__(self, idx: int):
        split_group = self.dataset_file[self.split]
        image = split_group[f"img_{idx}"][:]
        return torch.from_numpy(image), torch.scalar_tensor(1).long()

    def __del__(self):
        self.dataset_file.close()


class Negatives(Dataset):
    def __init__(self, root_path: str, split: str = "train") -> None:
        self.root_path = root_path
        self.split = split
        self.dataset_file = h5py.File(self.root_path, "r")

    def __len__(self) -> int:
        return len(self.dataset_file[self.split]["free_points"])

    def __getitem__(self, idx: int):
        split_group = self.dataset_file[self.split]
        item = split_group["free_points"][f"item_{idx}"]
        rand_idx = random.randint(0, item.shape[0] - 1)
        y, x = item[rand_idx]
        box_size, image_path = item.attrs["box_size"], item.attrs["image_path"]
        image = split_group[f"images/{image_path}"][:, y : y + box_size, x : x + box_size]
        return torch.from_numpy(image), torch.scalar_tensor(0).long()

    def __del__(self):
        self.dataset_file.close()


class ThermalPersonClassification(Dataset):
    def __init__(self, positives_path: str, negatives_path: str, split: str = "train") -> None:
        self.positives = Positives(root_path=positives_path, split=split)
        self.negatives = Negatives(root_path=negatives_path, split=split)
        self.total_size = len(self.positives) + len(self.negatives)

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int):
        if idx < len(self.positives):
            return self.positives.__getitem__(idx)
        return self.negatives.__getitem__(idx - len(self.positives))
