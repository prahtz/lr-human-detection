import json
import os
import torchvision
import pandas as pd
import h5py
import random
import torch

from torch.utils.data import Dataset

class Positives(Dataset):
    def __init__(self, root_path: str, split: str = 'train') -> None:
        self.split = split
        self.dataset_file = h5py.File(root_path, 'r')
        self.split_group = self.dataset_file[split]
        
    def __len__(self) -> int:
        return len(self.split_group)
    
    def __getitem__(self, idx: int):
        image = self.split_group[f'img_{idx}'][:]
        return torch.from_numpy(image)
    
    def __del__(self):
        self.dataset_file.close()

class Negatives(Dataset):
    def __init__(self, root_path: str, split: str = 'train') -> None:
        self.split = split
        self.dataset_file = h5py.File(root_path, 'r')
        self.split_group = self.dataset_file[split]
        
    def __len__(self) -> int:
        return len(self.split_group['free_points'])
    
    def __getitem__(self, idx: int):
        item = self.split_group['free_points'][f'item_{idx}']
        rand_idx = random.randint(0, item.shape[0] - 1)
        y, x = item[rand_idx]
        box_size, image_path = item.attrs['box_size'], item.attrs['image_path']
        image = self.split_group[f'images/{image_path}'][:, y:y+box_size, x:x+box_size]
        return torch.from_numpy(image)
    
    def __del__(self):
        self.dataset_file.close()