from typing import Tuple

import torch
import torch.nn as nn

from config.config import DatasetArgs, TestArgs, TrainingArgs
from datasets.data_modules import PRWClassificationDataModule, ThermalPersonClassificationDataModule


class BatchCollator:
    def __init__(self, transforms_fn):
        self.transforms_fn = transforms_fn

    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = [item[0] for item in batch], [item[1] for item in batch]
        inputs = [self.transforms_fn(tensor) for tensor in inputs]
        inputs = torch.stack(inputs)
        labels = torch.stack(labels).unsqueeze(-1).float()
        return inputs, labels


def load_data_module(
    data_args: DatasetArgs,
    training_args: TrainingArgs,
    test_args: TestArgs,
    train_transforms_fn: nn.Module,
    eval_transforms_fn: nn.Module,
):
    if data_args.dataset_name == "thermal-person-classification":
        data_module = ThermalPersonClassificationDataModule(
            data_args=data_args,
            training_args=training_args,
            test_args=test_args,
            train_transforms_fn=train_transforms_fn,
            eval_transforms_fn=eval_transforms_fn,
        )
    elif data_args.dataset_name == "prw-classification":
        data_module = PRWClassificationDataModule(
            data_args=data_args,
            training_args=training_args,
            test_args=test_args,
            train_transforms_fn=train_transforms_fn,
            eval_transforms_fn=eval_transforms_fn,
        )
    else:
        raise NotImplementedError("The specified dataset name is not supported.")

    return data_module
