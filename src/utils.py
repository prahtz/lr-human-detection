import math
import torch
import random
import numpy as np
import torch.distributed as dist
from typing import Optional, List
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler

class DistributedMultiplexerSampler(DistributedSampler):
    """
    Distributed sampler that samples from multiple map-style datasets according to the provided weights.
    Once a data source is exhausted, it gets randomly regenerated.

    Args:
        data_sources: List[Dataset]
            List of datasets where to sample indices.
        num_samples: Optional[int]
            Number of elements to be sampled. If None, it is equal to the total number of samples across data sources
        weights: Optional[List[float]]
            Weights assigned to each data source, proportional to its sampling probability. If None, it samples uniformly.
        rank: int
            Global process rank.
        num_replicas: int
            Total number of running processes.
        seed: int
            Starting random seed used to sample data sources and their elements. 
    """
    def __init__(self, data_sources: List[Dataset], 
                 num_samples: Optional[int] = None,
                 weights: Optional[List[float]] = None,
                 rank: int = 0,
                 num_replicas: int = 0,
                 seed: int = 0) -> None:
        self.data_sources = data_sources
        self.seed = seed
        self.rank, self.num_replicas = rank, num_replicas
        self.epoch = 0
        
        if num_samples is None:
            num_samples = sum([len(dataset) for dataset in data_sources])
        
        if weights is None:
            weights = [len(dataset) / num_samples for dataset in data_sources]

        self.num_samples = math.ceil(num_samples / self.num_replicas)
        self.weights = torch.tensor(weights)
        self.total_size = self.num_samples * self.num_replicas

        g = torch.Generator()
        g.manual_seed(self.seed)
        self.global_indices = [torch.randperm(len(dataset), generator=g).tolist() for dataset in self.data_sources]

        offsets = [0] + [len(dataset) for dataset in self.data_sources[:-1]]
        for i in range(1, len(offsets)):
            offsets[i] = offsets[i] + offsets[i-1]
        self.offsets = offsets

        self.source_gen = torch.Generator()
        self.source_gen.manual_seed(self.seed)

    def __iter__(self):    
        indices = []

        for _ in range(self.total_size):
            source_id = torch.multinomial(self.weights, 1, generator=self.source_gen)
            if not self.global_indices[source_id]:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                self.global_indices[source_id] = torch.randperm(len(self.data_sources[source_id]), generator=g).tolist()
            indices.append(self.global_indices[source_id].pop() + self.offsets[source_id])

        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
