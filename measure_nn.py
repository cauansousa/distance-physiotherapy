import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# 1- download dataset
# 2- create data loader
# 3- build model
# 4- train
# 5- save trained model

class measure(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.sequential(
            nn.Linear(4, ),
        )