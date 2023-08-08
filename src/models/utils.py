from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from config.config import ModelArgs, TestArgs
from models import custom_net, efficientnet
from models.model_modules import ThermalPersonClassificationModule


def load_model_and_transforms(model_args: ModelArgs, train: bool, checkpoint_path: Path | None = None, **kwargs):
    if model_args.name == "customnet":
        model, transforms_fn = (
            custom_net.load_customnet(),
            custom_net.get_customnet_transforms(train=train),
        )
    elif model_args.name == "efficientnet":
        model, transforms_fn = (
            efficientnet.load_efficientnet("efficientnet-b3"),
            efficientnet.get_efficientnet_transforms(train=train),
        )
    else:
        raise NotImplementedError("The specified model name is not supported")
    if checkpoint_path:
        module = ThermalPersonClassificationModule.load_from_checkpoint(checkpoint_path=checkpoint_path, model=model, map_location="cpu")
    else:
        module = ThermalPersonClassificationModule(model=model, **kwargs)
    return module, transforms_fn
