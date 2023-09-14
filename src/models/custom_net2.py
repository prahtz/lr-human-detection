import torch
import torchvision.transforms as transforms
from torch import nn


class CustomNet2(nn.Module):
    def __init__(self):
        super().__init__()

        max_pool2d = nn.MaxPool2d(2, 2)
        relu = nn.ReLU()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3)),
            nn.BatchNorm2d(num_features=16),
            relu,
            max_pool2d,
            nn.Conv2d(16, 32, (3, 3)),
            nn.BatchNorm2d(num_features=32),
            relu,
            max_pool2d,
            nn.Conv2d(32, 64, (3, 3)),
            nn.BatchNorm2d(num_features=64),
            relu,
            max_pool2d,
            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(num_features=64),
            relu,
            max_pool2d,
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x: torch.Tensor):
        x = self.conv_net(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


def load_customnet2():
    model = CustomNet2()
    return model


def get_customnet2_transforms():
    train_transforms_fn = transforms.Compose(
        [
            transforms.RandomResizedCrop((64, 64), scale=(0.25, 1), antialias=None),
            transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    eval_transforms_fn = transforms.Compose(
        [
            transforms.Resize((64, 64), antialias=None),
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return train_transforms_fn, eval_transforms_fn
