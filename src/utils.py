import math
import os
import random
from collections import defaultdict
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from background_subtraction import BackgroundSubtraction
from numpy.typing import NDArray
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
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
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


class BackgroundSubtractionForDetection:
    def __init__(self, background_samples, target_shape=None, c_optimized=True) -> None:
        self.target_shape = target_shape
        self.c_optimized = c_optimized
        if self.target_shape:
            height, width = background_samples[0].shape[:2]
            self.ratios = (height / target_shape[0], width / target_shape[1])
            for i in range(len(background_samples)):
                background_samples[i] = self.preprocess_image(background_samples[i], target_shape)

        self.bkg_subtraction = BackgroundSubtraction(
            background_samples,
            threshold=8,
            area_filter_fn=lambda area: area >= 50,
            opening_k_shape=(3, 3),
            closing_k_shape=(9, 9),
            beta=0.1,
            alpha=0.1,
            handle_light_changes=True,
            c_optimized=c_optimized,
        )

    def preprocess_image(self, img, target_size=(640, 640)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if self.c_optimized:
            img = cv2.resize(src=img, dsize=target_size[::-1], interpolation=cv2.INTER_LINEAR).astype(np.float32)
        else:
            img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
            img = F.interpolate(img, size=target_size)[0][0].numpy()
        return img

    def scale_up_bbox(self, bbox, ratios):
        bbox = [b for b in bbox]
        bbox[0] = round(bbox[0] * ratios[1])
        bbox[1] = round(bbox[1] * ratios[0])
        bbox[2] = round(bbox[2] * ratios[1])
        bbox[3] = round(bbox[3] * ratios[0])
        return bbox

    def step(self, frame: NDArray):
        frame = self.preprocess_image(frame, target_size=self.target_shape)
        candidate_bboxes = self.bkg_subtraction.step(frame)
        if self.target_shape:
            candidate_bboxes = [self.scale_up_bbox(bbox, self.ratios) for bbox in candidate_bboxes]
        return candidate_bboxes


class CustomCocoBinaryAveragePrecision:
    """
    This class computes the AP for object localization exactly as the COCO evaluation framework, but allowing
    the use of a custom detection function to discriminate between TP and FP.
    """

    def __init__(
        self,
        detection_fn: Callable,
    ) -> None:
        self.detection_fn = detection_fn
        self.detection_thresholds = np.linspace(0.5, 0.95, 10).tolist()

    def __call__(self, preds: list[dict[str, list[list[int | float] | list[float]]]], targets: list[list[list[int | float]]]):
        aps = {}
        all_preds, all_targets = defaultdict(list), defaultdict(list)
        for prediction, target_boxes in zip(preds, targets):
            detected_boxes, confidence_scores = prediction["boxes"], prediction["scores"]
            for detection_threshold in self.detection_thresholds:
                used_targets = [False] * len(target_boxes)
                sorted_pairs = sorted([(score, box) for score, box in zip(confidence_scores, detected_boxes)], key=lambda x: x[0])[::-1]
                confidence_scores, detected_boxes = [item[0] for item in sorted_pairs], [item[1] for item in sorted_pairs]
                pred, target = [], []
                for i in range(len(detected_boxes)):
                    matched = False
                    for j in range(len(target_boxes)):
                        detection_score = self.detection_fn(detected_boxes[i], target_boxes[j])
                        if detection_score > detection_threshold:
                            if not used_targets[j]:
                                pred.append(confidence_scores[i])
                                target.append(1)
                                used_targets[j] = True
                                matched = True
                    if not matched:
                        pred.append(confidence_scores[i])
                        target.append(0)
                for used in used_targets:
                    if not used:
                        pred.append(0.0)
                        target.append(1)
                all_preds[detection_threshold] += pred
                all_targets[detection_threshold] += target
        for detection_threshold in self.detection_thresholds:
            all_preds[detection_threshold] = torch.tensor(all_preds[detection_threshold])
            all_targets[detection_threshold] = torch.tensor(all_targets[detection_threshold])
            if all_preds[detection_threshold].numel():
                ap = average_precision(preds=all_preds[detection_threshold], target=all_targets[detection_threshold], task="binary")
            else:
                ap = 0.0
            aps[f"AP@{detection_threshold}"] = ap

        return aps
