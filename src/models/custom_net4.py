import torch
import torchvision.transforms as transforms
from torch import nn


class CustomNet4(nn.Module):
    def __init__(self):
        super().__init__()
        relu = nn.ReLU()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3)),
            nn.Conv2d(16, 16, (2, 2), 2),
            nn.BatchNorm2d(num_features=16),
            relu,
            nn.Conv2d(16, 32, (3, 3)),
            nn.Conv2d(32, 32, (2, 2), 2),
            nn.BatchNorm2d(num_features=32),
            relu,
            nn.Conv2d(32, 64, (3, 3)),
            nn.Conv2d(64, 64, (2, 2), 2),
            nn.BatchNorm2d(num_features=64),
            relu,
        )
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x: torch.Tensor):
        x = self.conv_net(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


def load_customnet4():
    model = CustomNet4()
    return model


def get_customnet4_transforms():
    train_transforms_fn = transforms.Compose(
        [
            transforms.RandomResizedCrop((32, 32)),
            transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    eval_transforms_fn = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return train_transforms_fn, eval_transforms_fn
