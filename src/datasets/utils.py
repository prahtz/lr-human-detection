from typing import Tuple

import torch
import torch.nn as nn

from config.config import DatasetArgs, TrainingArgs, TestArgs
from datasets.data_modules import ThermalPersonClassificationDataModule


class BatchCollator:
    def __init__(self, transforms_fn):
        self.transforms_fn = transforms_fn

    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = [item[0] for item in batch], [item[1] for item in batch]
        inputs = [self.transforms_fn(tensor) for tensor in inputs]
        inputs = torch.stack(inputs)
        labels = torch.stack(labels).unsqueeze(-1).float()
        return inputs, labels


def load_data_module(data_args: DatasetArgs, training_args: TrainingArgs, test_args: TestArgs, transforms_fn: nn.Module):
    data_module = ThermalPersonClassificationDataModule(
        data_args=data_args,
        training_args=training_args,
        test_args=test_args,
        transforms_fn=transforms_fn,
    )
    return data_module
