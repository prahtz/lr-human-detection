import os
import torch
from datasets.thermal_person_classification import Negatives, Positives, ConcatDataset

from config.config import DatasetArgs
from typing import Tuple


class BatchCollator:
    def __init__(self, transforms_fn):
        self.transforms_fn = transforms_fn
    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = [item[0] for item in batch], [item[1] for item in batch]
        inputs = [self.transforms_fn(tensor) for tensor in inputs]
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        return inputs, labels
    
def load_dataset(data_args: DatasetArgs, split: str):
    positives_path = os.path.join(data_args.root_path, data_args.positives_relative_path)
    negatives_path = os.path.join(data_args.root_path, data_args.negatives_relative_path)

    negatives = Negatives(root_path=negatives_path, split=split)
    positives = Positives(root_path=positives_path, split=split)
    dataset = ConcatDataset([negatives, positives])
    return dataset