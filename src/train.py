import torch.distributed as dist
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import utils
import argparse
from config.config import get_default_cfg
import models.utils
import datasets.utils
from training.evaluator import BinaryClassificationEvaluator
from training.trainer import Trainer
from training.callbacks import EarlyStoppingCallback

def pipeline(args):
    cfg_path = args.cfg_path
    random_seed = args.random_seed

    cfg = get_default_cfg()
    cfg.merge_from_file(cfg_path)

    data_args = cfg.dataset
    model_args = cfg.model
    training_args = cfg.training

    utils.init_distributed()
    utils.set_seed(random_seed)
    local_rank, _, _ = utils.get_ranks_and_replicas()

    train_dataset = datasets.utils.load_dataset(data_args, split='train')
    eval_dataset = datasets.utils.load_dataset(data_args, split='valid')

    model, transforms_fn = models.utils.load_model_and_transforms(model_args)

    if torch.cuda.is_available():
        model.cuda()
        
    if dist.is_torchelastic_launched():
        device_ids = [local_rank] if torch.cuda.is_available() else None
        model = DDP(model, device_ids=device_ids)

    batch_collator = datasets.utils.BatchCollator(transforms_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.lr)

    evaluator = BinaryClassificationEvaluator()

    callbacks = [EarlyStoppingCallback('f1', patience=10)]

    trainer = Trainer(training_args=training_args,
                      model=model,
                      optimizer=optimizer,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      collate_fn=batch_collator,
                      evaluator=evaluator,
                      callbacks=callbacks)
    
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('cfg_path', help='Path of the YAML configuration file.')
    parser.add_argument('--random-seed', help='Manual random seed', default=42, type=int)

    args = parser.parse_args()

    pipeline(args)