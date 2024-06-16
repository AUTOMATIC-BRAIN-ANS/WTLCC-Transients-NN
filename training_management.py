from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from Utilities.losses import get_criterion
from Utilities.lr_strategies import get_lr_strategy
from Utilities.optimizers import get_optimizer
from Utilities.metrics import AVAILABLE_METRICS
from plotting_management import get_plot
from Utilities.training_utils import EarlyStop, RunningAverage



def train(experiment_path, model, training_loader, val_loader, device, config, global_config, scalers=None, dry_run=False):
    criterion = get_criterion(config)
    optimizer = get_optimizer(config)(model.parameters())
    scheduler = get_lr_strategy(config)
    loss_avg = RunningAverage()
    if config["early_stopping"]:
        early_stop = EarlyStop(config["early_stopping_params"]["patience"])
    else:
        early_stop = None
    # print(len(training_loader))
    # print(len(val_loader))

    wandb.watch(model, criterion, log=global_config["log_model_params"], log_freq=global_config["report_training_every_n_batches"])

    batch_ct = 0
    for epoch in tqdm(range(config["epochs"])):
        pbar = tqdm(enumerate(training_loader), leave=False, total=len(training_loader))
        loss_avg.reset()
        for batch_idx, dct in pbar:
            model.train()
            loss = train_batch(dct, device, model, optimizer, criterion, config, batch_idx, len(training_loader))
            
            if loss.isnan():
                break
            
            loss_avg(loss.item())

            pbar.set_postfix(loss=loss.item())
            
            batch_ct += 1
            if (batch_ct % global_config["report_training_every_n_batches"]) == 0:
                # if config["accumulate_grads"]:
                #     metrics = {"training_loss": loss * config["batch_acumulator"]}
                # else:
                
                metrics = {"training_loss": loss}
                log(metrics, epoch, batch_ct)
            
            if dry_run:
                break
        
        if config["early_stopping"]:
            if early_stop(model, loss_avg.avg, epoch):
                break
        
        if global_config["log_training_metrics"] and epoch % global_config["log_training_stride"] == 0:
            val_metrics = validate(model, training_loader, device, config, "training", scalers[0])
            log(val_metrics, epoch, batch_ct)

        if global_config["validate_every_n_epochs"] and epoch % global_config["validation_stride"] == 0:
            if scalers is not None:
                val_metrics = validate(model, val_loader, device, config, "validation", scalers[0])
            else:
                val_metrics = validate(model, val_loader, device, config, "validation")
            log(val_metrics, epoch, batch_ct)
        plot_log_at_end = global_config["log_plots_at_end"] and (epoch + 1) == config["epochs"]
        plot_log_interval = global_config["log_additional_plots"] and epoch % global_config["plot_log_stride"] == 0
        if plot_log_at_end or plot_log_interval:
            plots = get_plot(model, training_loader, val_loader, epoch, device, config, global_config, scalers)
            log(plots, epoch, batch_ct)

        
        if scheduler is not None:
            scheduler.step()

        if global_config["checkpointing"] and epoch % global_config["checkpoint_every"] == 0:
            save(experiment_path, model, "checkpoint", optimizer=optimizer, 
                scheduler=scheduler, batch_no=batch_ct)

        if dry_run:
            break

    if global_config["save_on_training_end"]:
        save(experiment_path, model, global_config["save_mode"], 
            dct["data"], optimizer, scheduler)

    if dry_run:
        print("Finished checking model")
        return


def log(metrics, epoch, batch):
    metrics["epoch"] = epoch
    wandb.log(metrics, step=batch)


def validate(model, loader, device, config, prefix="validation", scaler=None, early_stop=None):
    model.eval()
    with torch.no_grad():
        all_outputs = torch.Tensor()
        all_labels = torch.Tensor()
        logging_dict = {}
        for dct in loader:
            if isinstance(dct["label"], dict):
                for key in dct["label"]:
                    if scaler is not None:
                        l_shape = dct["label"][key].shape
                        dct["label"][key] = torch.tensor(scaler.inverse_transform(dct["label"][key].reshape(-1, 1)).reshape(l_shape))
                    dct["label"][key] = dct["label"][key].to(device)
                labels = dct["label"]
            elif dct["label"] is None:
                labels = dct["label"]
            else:
                if scaler is not None:
                    label_shape = dct["label"].shape
                    dct["label"] = torch.tensor(scaler.inverse_transform(dct["label"].reshape(-1, 1)).reshape(label_shape))
                labels = dct["label"].to(device)
            
            if isinstance(dct["data"], dict):
                for key in dct["data"]:
                    if isinstance(dct["data"][key], dict):
                        for subkey in dct["data"][key]:
                            dct["data"][key][subkey] = dct["data"][key][subkey].float().to(device)
                    else:
                        dct["data"][key] = dct["data"][key].float().to(device)
                net_input = dct["data"]
            else:
                net_input = dct["data"].float().to(device)
            
            # outputs = model(net_input, "val")
            outputs = model(net_input)

            if scaler is not None:
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                output_shape = outputs.shape
                outputs = scaler.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1)).reshape(output_shape)
                outputs = torch.tensor(outputs).to(device)
            

            if isinstance(outputs, torch.Tensor):
                all_outputs = torch.cat((all_outputs, outputs.detach().cpu()), dim=0)
            elif isinstance(outputs, dict):
                if not isinstance(all_outputs, dict):
                    all_outputs = {}
                for key in outputs:
                    if key in all_outputs:
                        if "loss" in key or "MAE" in key:
                            all_outputs[key] = torch.cat((all_outputs[key], outputs[key].detach().cpu().reshape(1,-1)), dim=0)
                    elif "loss" in key or "MAE" in key:
                        all_outputs[key] = outputs[key].detach().cpu().reshape(1,-1)
            else:
                all_outputs = torch.cat((all_outputs, outputs[0].detach().cpu()), dim=0)
            if labels is not None:
                all_labels = torch.cat((all_labels, labels.detach().cpu()), dim=0)

        for metric in config["logging_metrics"]:
            val = AVAILABLE_METRICS[metric](all_outputs, all_labels, info={"patient_list": loader.dataset.patients})
            if isinstance(val, dict):
                for key in val:
                    logging_dict[prefix + "_" + key] = val[key]
            else:
                logging_dict[prefix + "_" + metric] = val
        return logging_dict

    


def save(experiment_path, model, mode, dummy_input=None, optimizer=None, scheduler=None, batch_no=None):
    output_ext = {
        "onnx": ".onnx",
        "torch": ".pth",
        "checkpoint": ".pth"
    }
    m_name = "net_model"
    if batch_no is not None:
        m_name += "_at_batch_{}".format(batch_no)
    m_name += output_ext[mode]
    save_path = experiment_path / m_name

    if mode == 'onnx':
        torch.onnx.export(model, dummy_input, save_path)
        wandb.save(str(save_path))
    if mode == 'torch':
        torch.save(model.state_dict(), save_path)
    if mode == 'checkpoint' and optimizer is not None:
        if scheduler is not None:
            torch.save(dict(model=model.state_dict(), 
                            optimizer=optimizer.state_dict(), 
                            scheduler=scheduler.state_dict()), 
                        save_path)
        else:
            torch.save(dict(model=model.state_dict(), 
                            optimizer=optimizer.state_dict(), 
                            scheduler=None), 
                        save_path)


def train_batch(dct, device, model, optimizer, criterion, config, batch_idx, max_batches):
    labels = dct["label"]
    if isinstance(labels, dict):
        for key in labels:
            if isinstance(labels[key], dict):
                for subkey in labels[key]:
                    labels[key][subkey] = labels[key][subkey].to(device)
            else:
                labels[key] = labels[key].to(device)
    elif labels is not None:
        labels = labels.to(device)
    
    net_input = dct["data"]
    if isinstance(net_input, dict):
        for key in net_input:
            if isinstance(net_input[key], dict):
                for subkey in net_input[key]:
                    net_input[key][subkey] = net_input[key][subkey].float().to(device)
            else:
                net_input[key] = net_input[key].float().to(device)
    elif isinstance(net_input, torch.Tensor):
        net_input = net_input.float().to(device)
    
    # outputs = model(net_input, "train")
    outputs = model(net_input)
    loss = criterion(outputs, labels)

    current_batch_is_update = False

    if config["accumulate_grads"]:
        # loss = loss / config["batch_acumulator"]
        current_batch_is_update = ((batch_idx + 1) % config["batch_acumulator"] == 0) or \
                 (batch_idx + 1 == max_batches)

    loss.backward()
    
    if not config["accumulate_grads"] or current_batch_is_update:
        if config["clip_grads"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_value"])
        optimizer.step()
        optimizer.zero_grad()

    return loss
