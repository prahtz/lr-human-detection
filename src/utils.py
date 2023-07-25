import math
import os
import random
from typing import Dict, Union

import numpy as np
import torch
import torch.distributed as dist


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
        backend = "gloo"
        if torch.cuda.is_available():
            backend = "nccl"
        dist.init_process_group(backend)
        local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)


def get_distributed_info():
    """
    Returns the local rank, the global rank and the number of running processes.
    """
    info = {
        "local_rank": 0,
        "global_rank": 0,
        "local_world_size": 1,
        "global_world_size": 1,
        "num_nodes": 1,
    }
    if dist.is_torchelastic_launched():
        info["local_rank"] = int(os.environ["LOCAL_RANK"])
        info["global_rank"] = int(os.environ["RANK"])
        info["local_world_size"] = int(os.environ["LOCAL_WORLD_SIZE"])
        info["global_world_size"] = int(os.environ["WORLD_SIZE"])
        info["num_nodes"] = math.ceil(
            info["global_world_size"] / info["local_world_size"]
        )
    return info


def is_main_process():
    rank = 0
    if dist.is_torchelastic_launched():
        rank = int(os.environ["RANK"])
    return rank == 0


def barrier():
    if dist.is_torchelastic_launched():
        dist.barrier()


def compute_iou(bb1: Dict[str, Union[float, int]], bb2: Dict[str, Union[float, int]]):
    area1, area2 = bb1["w"] * bb1["h"], bb2["w"] * bb2["h"]
    dx = min(bb1["x"] + bb1["w"], bb2["x"] + bb2["w"]) - max(bb1["x"], bb2["x"])
    dy = min(bb1["y"] + bb1["h"], bb2["y"] + bb2["h"]) - max(bb1["y"], bb2["y"])
    area_intersection = dx * dy if dx >= 0 and dy >= 0 else 0
    return area_intersection / (area1 + area2 - area_intersection)
