
import h5py
import random
import torch

from torch.utils.data import Dataset
from typing import List, Tuple
import torchvision.transforms as transforms

class Positives(Dataset):
    def __init__(self, root_path: str, split: str = 'train') -> None:
        self.root_path = root_path
        self.split = split
        with h5py.File(self.root_path, 'r') as dataset_file:
            self.len = len(dataset_file[split])   
        
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, idx: int):
        with h5py.File(self.root_path, 'r') as dataset_file:
            split_group = dataset_file[self.split]
            image = split_group[f'img_{idx}'][:]
            return torch.from_numpy(image), torch.scalar_tensor(1).long()

class Negatives(Dataset):
    def __init__(self, root_path: str, split: str = 'train') -> None:
        self.root_path = root_path
        self.split = split
        with h5py.File(self.root_path, 'r') as dataset_file:
            self.len = len(dataset_file[split]['free_points'])
        
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, idx: int):
        with h5py.File(self.root_path, 'r') as dataset_file:
            split_group = dataset_file[self.split]
            item = split_group['free_points'][f'item_{idx}']
            rand_idx = random.randint(0, item.shape[0] - 1)
            y, x = item[rand_idx]
            box_size, image_path = item.attrs['box_size'], item.attrs['image_path']
            image = split_group[f'images/{image_path}'][:, y:y+box_size, x:x+box_size]
            return torch.from_numpy(image), torch.scalar_tensor(0).long()

class ConcatDataset(Dataset):
    def __init__(self, data_sources: List[Dataset]):
        self.data_sources = data_sources
        self.total_size = sum([len(dataset) for dataset in self.data_sources])
        self.map_idx = [(i, j) for i, dataset in enumerate(data_sources) for j in range(len(dataset))]

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        source_id, idx = self.map_idx[idx]
        return self.data_sources[source_id].__getitem__(idx)


class BatchCollator:
    def __init__(self, transforms_fn):
        self.transforms_fn = transforms_fn
    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = [item[0] for item in batch], [item[1] for item in batch]
        inputs = [self.transforms_fn(tensor) for tensor in inputs]
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        return inputs, labels
    
def load_batch_collator():
    transforms_fn = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.Lambda(lambda x: x.float()),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
    return BatchCollator(transforms_fn)