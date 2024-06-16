from Models.InitialCNN import InitialModel
from Models.ViT import ViT

AVAILABLE_MODELS = {
    "CNN": InitialModel,
    "ViT": ViT
}

def get_model(config:dict, global_config:dict):
    model = AVAILABLE_MODELS[config["architecture"]]
    return model(**config["params"])