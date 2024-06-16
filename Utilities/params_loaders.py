import yaml
import os
from pathlib import Path

GLOBAL_PARAMS_PATH = Path("global_params.yaml")

def load_params(name:str) -> dict:
    with open(name, "r") as f:
        params = yaml.safe_load(f)

    return params

def load_experiment_params(name:str, project_name:str) -> dict:
    params_path = Path(os.getcwd()) / "Experiments" / project_name / "project_params.yaml" 
    all_params = load_params(params_path)
    
    return all_params[name]

def load_prevexperiment_params(run_path) -> dict:
    #TODO
    params_path = Path(run_path) / "all_params.yaml" 
    all_params = load_params(params_path)
    
    return all_params

class Dict2Class(object):
    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])
