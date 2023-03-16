import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from config.config import TrainingArgs

import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from tqdm.auto import tqdm

from torch import nn, optim
import utils

from utils import DistributedSequentialSampler

from typing import List, Callable


class Trainer:
    def __init__(self, 
                 training_args: TrainingArgs,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 train_dataset: Dataset, 
                 eval_dataset: Dataset,
                 collate_fn: Callable
                ):
        
        self.training_args = training_args
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn
        self.global_rank, self.local_rank, self.num_replicas = utils.get_ranks_and_replicas()

        self.train_loader = self.get_train_loader()
        self.eval_loader = self.get_eval_loader()

        if self.global_rank == 0:
            self.training_writer = SummaryWriter(training_args.log.log_path)

        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        training_args = self.training_args
        training_log = defaultdict(float)
        if self.global_rank == 0:
            total_updates = training_args.num_epochs * len(self.train_loader)
            self.training_progress_bar = tqdm(total=total_updates)

        for epoch in range(training_args.num_epochs):
            self.training_step(training_log)
            self.evaluation_step(training_log)
            self.log_results(training_log, epoch)
            
    def training_step(self, training_log):
        self.model.train()
        self.optimizer.zero_grad()
        training_log['train/loss'] = 0.0
        for batch_idx, data in enumerate(self.train_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
            logits = self.model(inputs)
            loss = self.loss_fn(logits, labels) / self.training_args.accumulation_steps
            loss.backward()
            if ((batch_idx + 1) % self.training_args.accumulation_steps == 0) or (batch_idx + 1 == len(self.train_loader)):
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.global_rank == 0:
                    self.training_progress_bar.update(1)

            training_log['train/loss'] += (loss.detach() / len(self.train_loader))

    def evaluation_step(self, training_log):
        self.model.eval()
        with torch.no_grad():
            training_log['eval/loss'] = 0.0
            training_log['eval/accuracy'] = 0.0
            for batch_idx, data in enumerate(self.eval_loader):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')
                logits = self.model(inputs)
                loss = self.loss_fn(logits, labels) / self.training_args.accumulation_steps
                preds = torch.argmax(logits, dim=-1)
                
                training_log['eval/loss'] += (loss.detach() / len(self.eval_loader))
                training_log['eval/accuracy'] += (torch.sum(preds == labels) / preds.shape[0]) / len(self.eval_loader)

    def log_results(self, training_log, epoch: int):
        for metric_name, metric_tensor in training_log.items():
            if dist.is_torchelastic_launched():
                tensor_list = utils.gather_tensor(metric_tensor, self.global_rank, self.num_replicas, 0)
                if self.global_rank == 0:
                    metric = torch.mean(torch.stack(tensor_list)).item()
                    self.training_writer.add_scalar(metric_name, metric, epoch)
            else:
                self.training_writer.add_scalar(metric_name, metric_tensor, epoch)

    def get_train_loader(self):
        train_sampler = DistributedSampler(dataset=self.train_dataset,
                                           num_replicas=self.num_replicas,
                                           rank=self.global_rank,
                                           shuffle=True,
                                           drop_last=False)
        train_loader = DataLoader(dataset=self.train_dataset, 
                                  batch_size=self.training_args.train_batch_size, 
                                  num_workers=self.training_args.num_workers, 
                                  collate_fn=self.collate_fn, 
                                  sampler=train_sampler)
        return train_loader
    
    def get_eval_loader(self):
        eval_sampler = DistributedSequentialSampler(dataset=self.eval_dataset,
                                                    num_replicas=self.num_replicas,
                                                    rank=self.global_rank,
                                                    shuffle=False,
                                                    drop_last=False)
        eval_loader = DataLoader(dataset=self.eval_dataset,
                                 batch_size=self.training_args.eval_batch_size,
                                 num_workers=self.training_args.num_workers, 
                                 collate_fn=self.collate_fn, 
                                 shuffle=False,
                                 sampler=eval_sampler)
        return eval_loader
    
    def gather_eval_predictions(self, preds: List[torch.Tensor], labels: List[torch.Tensor]):
        pass