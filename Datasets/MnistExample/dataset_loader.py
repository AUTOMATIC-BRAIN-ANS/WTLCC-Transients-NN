from typing import Callable, Optional
import torchvision
from Utilities.transforms import get_transforms

class MnistDataset(torchvision.datasets.MNIST):
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return {"data": img, "label": target}


def get_mnist(train, config):
    transform = get_transforms(config)
    full_dataset = MnistDataset(root="Datasets/MnistExample/Data",
                                                train=train, 
                                                transform=transform,
                                                download=config["download"])
    return full_dataset
