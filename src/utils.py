import math
import os
import random
from typing import Any, Callable, Dict, Union

import numpy as np
import torch
import torch.distributed as dist
from torchmetrics.functional import average_precision


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
        info["num_nodes"] = math.ceil(info["global_world_size"] / info["local_world_size"])
    return info


def is_main_process():
    rank = 0
    if dist.is_torchelastic_launched():
        rank = int(os.environ["RANK"])
    return rank == 0


def barrier():
    if dist.is_torchelastic_launched():
        dist.barrier()


def compute_intersection(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2

    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
    if dx < 0 or dy < 0:
        return 0.0
    return dx * dy


def compute_iou(bb1, bb2):
    area_intersection = compute_intersection(bb1, bb2)
    area1, area2 = bb1[2] * bb1[3], bb2[2] * bb2[3]
    return area_intersection / (area1 + area2 - area_intersection)


def compute_intersection_over_target_area(box, target_box):
    area_intersection = compute_intersection(box, target_box)
    target_area = target_box[2] * target_box[3]
    return area_intersection / target_area


class CustomCocoBinaryAveragePrecision:
    """
    This class computes the AP for object localization exactly as the COCO evaluation framework, but allowing
    the use of a custom detection function to discriminate between TP and FP.
    """

    def __init__(
        self,
        detection_fn: Callable,
        detection_thresholds: list[float] = [0.5],
    ) -> None:
        self.detection_fn = detection_fn
        self.detection_thresholds = detection_thresholds

    def __call__(self, detected_boxes: list[int | float], confidence_scores: list[float], target_boxes: list[int | float]) -> Any:
        aps = {}
        for detection_threshold in self.detection_thresholds:
            used_targets = [False] * len(target_boxes)
            sorted_pairs = sorted([(score, box) for score, box in zip(confidence_scores, detected_boxes)], key=lambda x: x[0])[::-1]
            confidence_scores, detected_boxes = [item[0] for item in sorted_pairs], [item[1] for item in sorted_pairs]
            preds, target = [], []
            for i in range(len(detected_boxes)):
                matched = False
                for j in range(len(target_boxes)):
                    detection_score = self.detection_fn(detected_boxes[i], target_boxes[j])
                    if detection_score > detection_threshold:
                        if not used_targets[j]:
                            preds.append(confidence_scores[i])
                            target.append(1)
                            used_targets[j] = True
                            matched = True
                if not matched:
                    preds.append(confidence_scores[i])
                    target.append(0)
            for used in used_targets:
                if not used:
                    preds.append(0.0)
                    target.append(1)

            preds, target = torch.tensor(preds), torch.tensor(target)
            ap = average_precision(preds=preds, target=target, task="binary")
            aps[f"AP@{detection_threshold}"] = ap

        return aps
