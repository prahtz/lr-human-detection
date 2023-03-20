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

from training.utils import DistributedSequentialSampler, gather_eval_predictions
from torch.utils.data import DataLoader

from typing import Union, List

from tqdm import tqdm

def pipeline(args):
    cfg_path = args.cfg_path
    random_seed = args.random_seed

    cfg = get_default_cfg()
    cfg.merge_from_file(cfg_path)

    data_args = cfg.dataset
    model_args = cfg.model
    test_args = cfg.test
    training_args = cfg.training

    utils.init_distributed()
    utils.set_seed(random_seed)
    local_rank, global_rank, num_replicas = utils.get_ranks_and_replicas()

    test_dataset = datasets.utils.load_dataset(data_args, split='test')

    model, transforms_fn = models.utils.load_model_from_training_state(model_args, path=test_args.training_state_path)

    if torch.cuda.is_available():
        model.cuda()
        
    if dist.is_torchelastic_launched():
        device_ids = [local_rank] if torch.cuda.is_available() else None
        model = DDP(model, device_ids=device_ids)

    batch_collator = datasets.utils.BatchCollator(transforms_fn)
    test_sampler = DistributedSequentialSampler(dataset=test_dataset,
                                                num_replicas=num_replicas,
                                                rank=global_rank,
                                                shuffle=False,
                                                drop_last=False)
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=test_args.test_batch_size,
                                num_workers=test_args.num_workers, 
                                collate_fn=batch_collator, 
                                shuffle=False,
                                sampler=test_sampler)

    evaluator = BinaryClassificationEvaluator()

    metrics = {}
    samples = [evaluation_step(model, evaluator, test_loader, test_dataset) for _ in range(training_args.eval_num_repetitions)]
    keys = samples[0].keys()
    means = {k: torch.mean(torch.tensor([s[k] for s in samples])).item() for k in keys}
    if training_args.eval_num_repetitions > 1:
        stds = {k: torch.std(torch.tensor([s[k] for s in samples])).item() for k in keys}
    else:
        stds = {k: 0.0 for k in keys}
    for k in keys:
        metrics[f'test/{k}'] = means[k]
        metrics[f'test/std_{k}'] = stds[k]
    if global_rank == 0:
        print(metrics)

def evaluation_step(model, evaluator, loader, test_dataset):
    model.eval()
    logits_list, label_list = [], []
    _, rank, num_replicas = utils.get_ranks_and_replicas()
    if rank == 0:
        print('Running Evaluation...')
        progress_bar = tqdm(total=len(loader))
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
            logits = model(inputs)

            logits_list.append(logits.detach())
            label_list.append(labels.detach())

            if rank == 0:
                progress_bar.update(1)
    
        logits, labels = gather_eval_predictions(logits_list, label_list, num_replicas)
        logits = logits[:len(test_dataset)]
        labels = labels[:len(test_dataset)]
        metrics = evaluator(logits, labels)
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('cfg_path', help='Path of the YAML configuration file.')
    parser.add_argument('--random-seed', help='Manual random seed', default=42, type=int)

    args = parser.parse_args()

    pipeline(args)