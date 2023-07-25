import os

import lightning as L
import torch.nn as nn
from torch.utils.data import DataLoader

import datasets.utils
from config.config import DatasetArgs, TrainingArgs
from datasets.thermal_person_classification import ThermalPersonClassification


class ThermalPersonClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_args: DatasetArgs,
        training_args: TrainingArgs,
        transforms_fn: nn.Module,
    ):
        super().__init__()
        self.data_args = data_args
        self.training_args = training_args
        self.transforms_fn = transforms_fn

    def setup(self, stage: str):
        positives_path = os.path.join(
            self.data_args.root_path, self.data_args.positives_relative_path
        )
        negatives_path = os.path.join(
            self.data_args.root_path, self.data_args.negatives_relative_path
        )

        self.datasets = {}
        for split in ("train", "valid"):
            self.datasets[split] = ThermalPersonClassification(
                positives_path=positives_path,
                negatives_path=negatives_path,
                split=split,
            )
        self.collate_fn = datasets.utils.BatchCollator(self.transforms_fn)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.training_args.train_batch_size,
            num_workers=self.training_args.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.datasets["valid"],
            batch_size=self.training_args.eval_batch_size,
            num_workers=self.training_args.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
        )


def load_data_module(
    data_args: DatasetArgs, training_args: TrainingArgs, transforms_fn: nn.Module
):
    data_module = ThermalPersonClassificationDataModule(
        data_args=data_args, training_args=training_args, transforms_fn=transforms_fn
    )

    return data_module
