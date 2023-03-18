import os
import math
import torch
import random
import numpy as np
import torch.distributed as dist
from typing import List

def set_seed(seed: int):
    """
    Initializes the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_distributed():
    """
    Initializes the distributed environment.
    """
    if dist.is_torchelastic_launched():
        backend = 'gloo'
        if torch.cuda.is_available():
            backend = 'nccl'
        dist.init_process_group(backend)
        local_rank = int(os.environ['LOCAL_RANK'])
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

def get_ranks_and_replicas():
    """
    Returns the local rank, the global rank and the number of running processes.
    """
    local_rank, global_rank, num_replicas = 0, 0, 1
    if dist.is_torchelastic_launched():
        local_rank, global_rank, num_replicas = int(os.environ['LOCAL_RANK']), int(os.environ['RANK']), dist.get_world_size()
    return local_rank, global_rank, num_replicas

def is_main_process():
    rank = 0
    if dist.is_torchelastic_launched():
        rank = int(os.environ['RANK'])
    return rank == 0

def barrier():
    if dist.is_torchelastic_launched():
        dist.barrier()

def gather_tensor(tensor, rank, num_replicas, dst_rank):
    """
    Wraps the distributed gather function by automatically creating the tensor list.
    """
    tensor_list = [tensor for _ in range(num_replicas)] if rank == dst_rank else None
    dist.gather(tensor, tensor_list, dst=0)
    return tensor_list

def gather_object(obj, rank, num_replicas, dst_rank):
    obj_list = [obj for _ in range(num_replicas)] if rank == dst_rank else None
    dist.gather_object(obj, obj_list, dst=0)
    return obj_list

def all_gather(tensor: torch.Tensor, num_replicas):
    tensor_list = [tensor.clone().to(tensor.device) for _ in range(num_replicas)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list

def all_gather_object(obj, num_replicas):
    obj_list = [obj for _ in range(num_replicas)]
    dist.all_gather_object(obj_list, obj)
    return obj_list

def all_gather_tensor(tensor: torch.Tensor, num_replicas: int, pad_value=-100, pad_dim: int = -1):
    max_len = max(all_gather(torch.tensor(tensor.shape[pad_dim], device=tensor.device), num_replicas))
    tensor = torch.nn.functional.pad(tensor, get_padding_tuple(tensor, max_len - tensor.shape[pad_dim], dim=pad_dim), 'constant', pad_value)
    tensor_list = [tensor.clone() for _ in range(num_replicas)]
    dist.all_gather(tensor_list, tensor)
    output_tensor = torch.cat(tensor_list, dim=0)
    return output_tensor

def get_padding_tuple(tensor: torch.Tensor, padding_size: int, dim: int):
    pad = [0 for _ in range(2*tensor.dim())]
    pad[2*dim] = padding_size
    pad = pad[::-1]
    return pad 

def pad_tensor_list(tensor_list: List[torch.Tensor], pad_value: int, dim: int = -1, max_len=None):
    dim = tensor_list[0].dim() + dim if dim < 0 else dim
    if max_len is None:
        max_len = max([t.shape[dim] for t in tensor_list])
    
    for i in range(len(tensor_list)):
        pad = get_padding_tuple(tensor_list[i], max_len - tensor_list[i].shape[dim], dim=dim)
        tensor_list[i] = torch.nn.functional.pad(tensor_list[i], pad, 'constant', pad_value)
    return tensor_list

def pad_stack(tensor_list: List[torch.Tensor], pad_value: int, pad_dim: int = -1):
    return torch.stack(pad_tensor_list(tensor_list, pad_value=pad_value, dim=pad_dim), dim=0)

def pad_cat(tensor_list: List[torch.Tensor], pad_value: int, pad_dim: int = -1):
    return torch.cat(pad_tensor_list(tensor_list, pad_value=pad_value, dim=pad_dim), dim=0)