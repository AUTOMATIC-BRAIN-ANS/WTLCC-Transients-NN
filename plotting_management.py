from random import randint
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import wandb

def plot_random_datapoint(model, data, device, scalers):
    random_batch = next(iter(data))
    random_data = random_batch["data"]
    dict_flag = False
    vae_flag = False
    if isinstance(random_data, dict):
        if "Raw" in random_data.keys():
            random_data = random_data["Raw"]
            dict_flag = True
            scalers = scalers[0], scalers[1]["Raw"]
        else:
            holdout = random_data["X_holdout"].to(device)
            mask = random_data["missing_mask"].to(device)
            random_data = random_data["X"]
            vae_flag = True
            scalers = None
    batch_size = random_data.shape[0]
    random_batch_index = randint(0, batch_size - 1)
    random_data = random_data[random_batch_index, :, :].unsqueeze(0).to(device)
    random_label = random_batch["label"][random_batch_index, :, :]
    if dict_flag:
        avg = random_batch["data"]["Averaged"][random_batch_index, :, :].unsqueeze(0).to(device)
        random_output = model({"Raw": random_data, "Averaged": avg})
    elif vae_flag:
        mask = mask[random_batch_index, :, :].unsqueeze(0)
        holdout = holdout[random_batch_index, :, :].unsqueeze(0)
        random_output = model({"X": random_data, "X_holdout":holdout, "missing_mask": mask})
        mask = mask.cpu().detach().numpy()
        holdout = holdout.cpu().detach().numpy()
    else:
        random_output = model(random_data)

    if isinstance(random_output, tuple):
        random_output = random_output[0]

    fig, axs = plt.subplots(2, 1, figsize=(21, 7))
    
    if scalers is not None:
        input_scaler = scalers[1]
        output_scaler = scalers[0]
        
        random_data_shape = random_data.shape
        random_data = input_scaler.inverse_transform(random_data.reshape(-1,2).cpu().detach().numpy()).reshape(random_data_shape)
        
        random_label_shape = random_label.shape
        random_label = output_scaler.inverse_transform(random_label.reshape(-1,1).cpu().detach().numpy()).reshape(random_label_shape)

        random_output_shape = random_output.shape
        random_output = output_scaler.inverse_transform(random_output.reshape(-1,1).cpu().detach().numpy()).reshape(random_output_shape)
    else:
        random_data = random_data.cpu().detach().numpy()
        random_label = random_label.cpu().detach().numpy()
        random_output = random_output.cpu().detach().numpy()
        
    if not vae_flag:
        axs[0].plot(random_data[0,:,0], label="ABP")
        axs[0].plot(random_data[0,:,1], label="Flow Velocity")
    else:
        axs[0].plot((random_data * mask)[0, :, 0], label="Input")
    axs[0].set_ylabel('Inputs')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(random_label[:], label="real")
    axs[1].plot(random_output[0,:,:], label="predicted")
    axs[1].set_ylabel('Output - real vs predicted')
    axs[1].grid(True)
    axs[1].legend()

    fig.tight_layout()
    return wandb.Image(fig)

def plot_toy_sig2sig(model, training_data, val_data, epoch, device, config, global_config, scalers=None):
    training_plots = global_config["number_of_training_set_plots"]
    val_plots = global_config["number_of_val_set_plots"]
    plots = {}
    for i in range(training_plots):
        fig = plot_random_datapoint(model, training_data, device, scalers)
        plots[f"sig2sig_training_set_plot_epoch_{epoch}_example_{i}"] = fig
    
    for i in range(val_plots):
        fig = plot_random_datapoint(model, val_data, device, scalers)
        plots[f"sig2sig_val_set_plot_epoch_{epoch}_example_{i}"] = fig
    plt.close("all")
    return plots
    

AVAILABLE_PLOTS = {
    "ToySig2Sig": plot_toy_sig2sig,
}

def get_plot(model, training_data, val_data, epoch, device, config, global_config, scalers):
    #Returns a dictionary of matplotlib figures to log for the model
    full_plot_dict = {}
    if config["plots_to_log"] is not None:
        for plot_name in config["plots_to_log"]:
            plots = AVAILABLE_PLOTS[plot_name](model, training_data, val_data, epoch, device, config, global_config, scalers)
            full_plot_dict.update(plots)
    return full_plot_dict