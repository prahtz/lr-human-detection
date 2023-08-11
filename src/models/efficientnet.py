import torchvision
import torchvision.transforms as transforms
from torch import nn
from torchvision.models import EfficientNet_B3_Weights


def load_efficientnet(model_name: str, freeze_last_layers: int = 4):
    if model_name == "efficientnet-b3":
        model = torchvision.models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False

        for param in model.features[-freeze_last_layers:].parameters():
            param.requires_grad = True

        model.classifier[1] = nn.Linear(in_features=1536, out_features=1)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    raise NotImplementedError("The name of the provided model is not supported")


def get_efficientnet_transforms():
    train_transforms_fn = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    eval_transforms_fn = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.3)),
        ]
    )
    return train_transforms_fn, eval_transforms_fn
