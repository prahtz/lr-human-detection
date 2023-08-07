import os

import lightning as L
import torch.nn as nn
from torch.utils.data import DataLoader

import datasets.utils
from config.config import DatasetArgs, TestArgs, TrainingArgs
from datasets.prw import PRWClassification
from datasets.thermal_person_classification import ThermalPersonClassification


class ThermalPersonClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_args: DatasetArgs,
        training_args: TrainingArgs,
        test_args: TestArgs,
        transforms_fn: nn.Module,
    ):
        super().__init__()
        self.data_args = data_args
        self.training_args = training_args
        self.test_args = test_args
        self.transforms_fn = transforms_fn

    def setup(self, stage: str):
        positives_path = os.path.join(self.data_args.root_path, self.data_args.positives_relative_path)
        negatives_path = os.path.join(self.data_args.root_path, self.data_args.negatives_relative_path)

        self.datasets = {}
        for split in ("train", "valid", "test"):
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
            shuffle=True,
        )

    def val_dataloader(self):
        dataloaders = [
            DataLoader(
                dataset=self.datasets["valid"],
                batch_size=self.training_args.eval_batch_size,
                num_workers=self.training_args.num_workers,
                collate_fn=self.collate_fn,
                shuffle=False,
            )
        ]
        for _ in range(self.training_args.eval_num_repetitions - 1):
            dataloaders.append(
                DataLoader(
                    dataset=self.datasets["valid"].negatives,
                    batch_size=self.training_args.eval_batch_size,
                    num_workers=self.training_args.num_workers,
                    collate_fn=self.collate_fn,
                    shuffle=False,
                )
            )
        return dataloaders

    def test_dataloader(self):
        dataloaders = [
            DataLoader(
                dataset=self.datasets["test"],
                batch_size=self.test_args.test_batch_size,
                num_workers=self.test_args.num_workers,
                collate_fn=self.collate_fn,
                shuffle=False,
            )
        ]
        for _ in range(self.test_args.test_num_repetitions - 1):
            dataloaders.append(
                DataLoader(
                    dataset=self.datasets["test"].negatives,
                    batch_size=self.test_args.test_batch_size,
                    num_workers=self.test_args.num_workers,
                    collate_fn=self.collate_fn,
                    shuffle=False,
                )
            )
        return dataloaders


class PRWClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_args: DatasetArgs,
        training_args: TrainingArgs,
        test_args: TestArgs,
        transforms_fn: nn.Module,
    ):
        super().__init__()
        self.data_args = data_args
        self.training_args = training_args
        self.test_args = test_args
        self.transforms_fn = transforms_fn

    def setup(self, stage: str):
        self.datasets = {}
        for split in ("train", "valid", "test"):
            self.datasets[split] = PRWClassification(
                root_path=self.data_args.root_path,
                split=split,
            )
        self.collate_fn = datasets.utils.BatchCollator(self.transforms_fn)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.training_args.train_batch_size,
            num_workers=self.training_args.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        dataloaders = [
            DataLoader(
                dataset=self.datasets["valid"],
                batch_size=self.training_args.eval_batch_size,
                num_workers=self.training_args.num_workers,
                collate_fn=self.collate_fn,
                shuffle=False,
            )
        ]
        for _ in range(self.training_args.eval_num_repetitions - 1):
            dataloaders.append(
                DataLoader(
                    dataset=self.datasets["valid"].negatives,
                    batch_size=self.training_args.eval_batch_size,
                    num_workers=self.training_args.num_workers,
                    collate_fn=self.collate_fn,
                    shuffle=False,
                )
            )
        return dataloaders

    def test_dataloader(self):
        dataloaders = [
            DataLoader(
                dataset=self.datasets["test"],
                batch_size=self.test_args.test_batch_size,
                num_workers=self.test_args.num_workers,
                collate_fn=self.collate_fn,
                shuffle=False,
            )
        ]
        for _ in range(self.test_args.test_num_repetitions - 1):
            dataloaders.append(
                DataLoader(
                    dataset=self.datasets["test"].negatives,
                    batch_size=self.test_args.test_batch_size,
                    num_workers=self.test_args.num_workers,
                    collate_fn=self.collate_fn,
                    shuffle=False,
                )
            )
        return dataloaders
