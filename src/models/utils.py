from models import custom_net
from models import efficientnet

from config.config import ModelArgs
def load_model_and_transforms(model_args: ModelArgs):
    if model_args.name == 'customnet':
        return custom_net.CustomNet(), custom_net.get_customnet_transforms()
    if model_args.name == 'efficientnet':
        return efficientnet.load_efficientnet('efficientnet-b3'), efficientnet.get_efficientnet_transforms()
    raise NotImplementedError('The specified model name is not supported')
    