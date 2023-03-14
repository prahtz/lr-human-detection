import torchvision
from torch import nn

def load_efficientnet(model_name: str):
    if model_name == 'efficientnet_b3':
        model = torchvision.models.efficientnet_b3()
        model.classifier[1] = nn.Linear(in_features=1536, out_features=2)
        return model
    raise NotImplementedError('The name of the provided model is not supported')