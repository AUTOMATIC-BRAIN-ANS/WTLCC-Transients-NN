
AVAILABLE_LR_STRATEGIES = {

}

def get_lr_strategy(config:dict):
    if config["lr_strategy"] is None:
        return None
    elif config["lr_params"] is None:
        return AVAILABLE_LR_STRATEGIES[config["lr_strategy"]]()
    else:
        return AVAILABLE_LR_STRATEGIES[config["lr_strategy"]](**config["lr_params"])