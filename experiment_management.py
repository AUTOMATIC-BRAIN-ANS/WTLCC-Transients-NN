import random
from pathlib import Path
import json
import yaml

import numpy as np
import torch
import wandb

from Utilities import params_loaders
import dataset_management
import model_management
import training_management
import sys
import argparse





def run_experiment(name: str, project: str, params_override_path:str = None, mode="online", sweep_params=None, dry_run=False):
    global_params = params_loaders.load_params(params_loaders.GLOBAL_PARAMS_PATH)
    experiment_params = params_loaders.load_experiment_params(name, project)
    
    experiment_data_path = Path("PreviousExperiments") / project / name
    experiment_data_path.mkdir(parents=True, exist_ok=True)
    
    directories = [int(f.name.split("_")[1]) for f in experiment_data_path.iterdir() if f.is_dir()]
    if len(directories) > 0:
        mx = max(directories)
    else:
        mx = 1
    experiment_data_path /= "Run_{}".format(mx+1)
    experiment_data_path.mkdir(parents=True, exist_ok=True)
    
    #Save experiment params
    with open(experiment_data_path/"all_params.yaml", "w") as f:
        yaml.dump({"global_params": global_params, "experiment_params": experiment_params}, f, default_flow_style=False)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(experiment_params["seed"])
    np.random.seed(experiment_params["seed"])
    torch.manual_seed(experiment_params["seed"])
    torch.cuda.manual_seed_all(experiment_params["seed"])

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() and global_params["global"]["gpu"] else "cpu")
    # Wandb login
    if mode != "disabled":
        wandb.login()

    # Experiment pipeline config
    config = experiment_params["config"]
    if sweep_params is not None:
        project = None
        name = None
        for key in sweep_params:
            vals = key.split(".")
            cnfg_ptr = config
            for subkey in vals[:-1]:
                cnfg_ptr = cnfg_ptr[subkey]
            cnfg_ptr[vals[-1]] = sweep_params[key]

    with wandb.init(entity="sonata-project", project=project, name=name, tags=experiment_params["tags"], 
                    notes=experiment_params["notes"], config=config, mode=mode):

        config = wandb.config

        # Make the datasts
        train_loader, scalers = dataset_management.get_data(config=config["dataset"], train=True, global_config=global_params["dataset"])
        test_loader, _ = dataset_management.get_data(config=config["dataset"], train=False, global_config=global_params["dataset"])
        
        # Make the model
        model = model_management.get_model(config=config["model"], global_config=global_params).to(device)
        if config["model"]["load_weights"]:
            model.load_state_dict(torch.load(Path(config["model"]["load_weights"])))
        # print(next(model.parameters()).device)
        # Train the model
        training_management.train(experiment_data_path, model, train_loader, test_loader,
                                device, config["training"], global_params["training"], scalers, dry_run=dry_run)

    return model



if __name__ == "__main__":
    #Mainly sweep purposes
    experiment = "BOTHCNN"
    project = "CorrMatToOutcome"
    args = sys.argv[1]
    with open(args, "r") as f:
        arg_dict = json.load(f)
        
    # if arg_dict["model.architecture"] == "ViT":
    #     experiment = "BOTHCNN"
    # else:
    #     experiment = "InitialCNN"

    run_experiment(experiment, project, sweep_params=arg_dict)