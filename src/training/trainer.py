import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from config.config import TrainingArgs

import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from tqdm import tqdm

from torch import nn, optim
import utils

from training.utils import DistributedSequentialSampler, gather_eval_predictions

from typing import List, Callable
from training.evaluator import Evaluator
from training.callbacks import Callback, TrainerControl


class Trainer:
    def __init__(self, 
                 training_args: TrainingArgs,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 train_dataset: Dataset, 
                 eval_dataset: Dataset,
                 collate_fn: Callable,
                 evaluator: Evaluator,
                 callbacks: List[Callback] = []
                ):
        
        self.training_args = training_args
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn
        self.evaluator = evaluator
        self.callbacks = callbacks
        self.local_rank, self.global_rank, self.num_replicas = utils.get_ranks_and_replicas()

        self.train_loader = self.get_train_loader()
        self.eval_loader = self.get_eval_loader()

        if self.global_rank == 0:
            self.training_writer = SummaryWriter(training_args.log.log_path)
        
        self.loss_fn = nn.CrossEntropyLoss()

        self.training_state_path = os.path.join(training_args.log.models_path, 'training_state.pkl')
        self.best_training_state_path = os.path.join(training_args.log.models_path, 'best_training_state.pkl')

        self.compare_metrics = lambda new, old: new > old if training_args.greater_is_better else lambda new, old: new > old
        self.best_metric = 0.0

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
            control = self.run_training_callbacks(training_log, epoch)

            if control.training_save:
                self.save_training_state()
            if control.training_stop:
                break

            if self.training_args.metric_for_best_model is not None:
                eval_metric = training_log[f'eval/{self.training_args.metric_for_best_model}']
                if epoch == 0 or self.compare_metrics(eval_metric, self.best_metric):
                    self.best_metric = eval_metric
                    self.save_training_state(self.best_training_state_path)

        if self.training_args.metric_for_best_model is not None:
            self.load_training_state(self.best_training_state_path)
        
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
        pred_list, label_list = [], []
        if self.global_rank == 0:
            print('Running Evaluation...')
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

                pred_list.append(preds.detach())
                label_list.append(labels.detach())
            
                training_log['eval/loss'] += (loss.detach() / len(self.eval_loader))
            
            for k, v in training_log.items():
                if dist.is_torchelastic_launched():
                    value_list = utils.all_gather_object(v, self.num_replicas)
                    training_log[k] = sum(value_list) / len(value_list)
            preds, labels = gather_eval_predictions(pred_list, label_list, self.num_replicas)
            metrics = self.evaluator(preds, labels)
            for k, v in metrics.items():
                training_log[f'eval/{k}'] = v

    def log_results(self, training_log, epoch: int):
        for metric_name, metric_value in training_log.items():
            if self.global_rank == 0:
                self.training_writer.add_scalar(metric_name, metric_value, epoch)

    def run_training_callbacks(self, training_log, epoch):
        control = TrainerControl()
        for callback in self.callbacks:
            current_control = callback(training_log, epoch)
            control.training_stop = control.training_stop or current_control.training_stop
            control.training_save = control.training_save or current_control.training_save
        return control
    
    def save_training_state(self, path=None):
        if path is None:
            path = self.training_state_path
        if self.global_rank == 0:
            training_state = {}
            training_state['optimizer_state_dict'] = self.optimizer.state_dict()
            training_state['model_state_dict'] = self.model.state_dict()
            torch.save(training_state, path)
        utils.barrier()

    def load_training_state(self, path=None):
        if path is None:
            path = self.training_state_path
        training_state = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(training_state['model_state_dict'])
        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])

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