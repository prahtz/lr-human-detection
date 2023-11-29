from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from config.config import ModelArgs
from models import custom_net, custom_net2, custom_net3, efficientnet
from models.model_modules import PRWClassificationModule


def load_model_and_transforms(model_args: ModelArgs, checkpoint_path: Path | None = None, **kwargs):
    if model_args.name == "customnet":
        model, transforms_fn = (
            custom_net.load_customnet(),
            custom_net.get_customnet_transforms(),
        )
    elif model_args.name == "customnet2":
        model, transforms_fn = (
            custom_net2.load_customnet2(),
            custom_net2.get_customnet2_transforms(),
        )
    elif model_args.name == "customnet3":
        model, transforms_fn = (
            custom_net3.load_customnet3(),
            custom_net3.get_customnet3_transforms(),
        )
    elif model_args.name == "efficientnet":
        model, transforms_fn = (
            efficientnet.load_efficientnet("efficientnet-b3"),
            efficientnet.get_efficientnet_transforms(),
        )
    else:
        raise NotImplementedError("The specified model name is not supported")

    if model_args.task == "classification":
        module_class = PRWClassificationModule
    else:
        raise NotImplementedError("The specified task is not supported")

    if checkpoint_path:
        module = module_class.load_from_checkpoint(checkpoint_path=checkpoint_path, model=model, map_location="cpu", strict=False)
    else:
        module = module_class(model=model, **kwargs)

    if len(transforms_fn) == 2:
        train_transforms_fn, eval_transforms_fn = transforms_fn
        label_transforms_fn = None
    else:
        train_transforms_fn, eval_transforms_fn, label_transforms_fn = transforms_fn
    return module, train_transforms_fn, eval_transforms_fn, label_transforms_fn
