import torch
from config.config import ModelArgs
from models import custom_net
from models import efficientnet
from collections import OrderedDict

def load_model_and_transforms(model_args: ModelArgs):
    if model_args.name == 'customnet':
        return custom_net.load_customnet(), custom_net.get_customnet_transforms()
    if model_args.name == 'efficientnet':
        return efficientnet.load_efficientnet('efficientnet-b3'), efficientnet.get_efficientnet_transforms()
    raise NotImplementedError('The specified model name is not supported')

def load_model_from_training_state(model_args: ModelArgs, path: str):
    training_state = torch.load(path, map_location=torch.device('cpu'))
    model, transforms_fn = load_model_and_transforms(model_args)
    state_dict = training_state['model_state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'module.' == k[:7]:
            k = k[7:]
        new_state_dict[k]=v

    model.load_state_dict(new_state_dict)
    return model, transforms_fn
    