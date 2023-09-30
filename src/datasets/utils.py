from typing import Tuple

import torch


class BatchCollator:
    def __init__(self, transforms_fn):
        self.transforms_fn = transforms_fn

    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = [item[0] for item in batch], [item[1] for item in batch]
        inputs = [self.transforms_fn(tensor) for tensor in inputs]
        inputs = torch.stack(inputs)
        labels = torch.stack(labels).unsqueeze(-1).float()
        return inputs, labels
