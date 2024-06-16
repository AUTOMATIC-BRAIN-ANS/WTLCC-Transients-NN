import torch

AVAILABLE_OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}

def get_optimizer(config:dict):
    lr = config["lr"]
    optimizer = AVAILABLE_OPTIMIZERS[config["optimizer"]]
    if config["optimizer_params"] is not None:    
        return lambda params: optimizer(params, lr=lr, **config["optimizer_params"])
    else:
        return lambda params: optimizer(params, lr=lr)
