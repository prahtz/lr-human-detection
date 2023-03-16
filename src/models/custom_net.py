import torch
from torch import nn
import torchvision.transforms as transforms

class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()

        max_pool2d = nn.MaxPool2d(2, 2)
        relu = nn.ReLU()
        self.conv_net = nn.Sequential(
                nn.Conv2d(3, 16, (3, 3)),
                relu,
                max_pool2d,
                nn.Conv2d(16, 32, (3, 3)),
                relu,
                max_pool2d,
                nn.Conv2d(32, 64, (3, 3)),
                relu,
                max_pool2d,
                nn.Conv2d(64, 64, (3, 3)),
                relu,
                max_pool2d,
                nn.Conv2d(64, 64, (3, 3)),
                relu,
                max_pool2d,
            )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256, out_features=2)
    
    def forward(self, x: torch.Tensor):
        x = self.conv_net(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


def load_customnet():
    model = CustomNet()
    return model

def get_customnet_transforms():
    transforms_fn = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.Lambda(lambda x: x.float()),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
    return transforms_fn