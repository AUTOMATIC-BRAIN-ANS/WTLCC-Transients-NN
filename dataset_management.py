# import torchvision
import numpy as np
import scipy.signal
from pathlib import Path
import torch
import os
from tqdm import tqdm
import math
from Datasets.MnistExample.dataset_loader import get_mnist
from Datasets.PickleLoader.dataset_loader import get_md
from Utilities.transforms import AVAILABLE_TRANSFORMS     

AVAILABLE_DATASETS = {
    "MNIST": get_mnist,
    "MatrixData": get_md
}


def get_data(train:bool, config:dict, global_config:dict):
    dataset = AVAILABLE_DATASETS[config["name"]](train, config)
    shuffle = global_config["shuffle"]
    if not train:
        shuffle = False
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=config["batch_size"], 
                                         shuffle=shuffle,
                                         pin_memory=True, 
                                         num_workers=global_config["dataset_num_workers"])
    if hasattr(dataset, "output_scaler"):
        return loader, (dataset.output_scaler, dataset.input_scaler)
    
    return loader, None
    
