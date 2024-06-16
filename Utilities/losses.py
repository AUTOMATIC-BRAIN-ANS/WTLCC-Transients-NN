import torch.nn as nn
import torch
import torch.nn.functional as F

class BalancedBCELoss:
    def __init__(self, class_weights=None) -> None:
        self.class_weights = class_weights
        self.bce_fcn = F.binary_cross_entropy

    def __call__(self, outputs, labels):
        if self.class_weights is not None:
            weight = torch.tensor([self.class_weights[0] if labels[i] == 0 else self.class_weights[1] for i in range(len(labels))]).to(outputs.device)
            return self.bce_fcn(outputs, labels, weight=weight)
        else:
            return self.bce_fcn(outputs, labels)


class InputationLoss:
    def __init__(self, model, consistency_loss_weight, reconstruction_loss_weight, imputation_loss_weight, MIT, ORT) -> None:
        self.model = model
        self.consistency_loss_weight = consistency_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.imputation_loss_weight = imputation_loss_weight
        self.MIT = MIT
        self.ORT = ORT
    
    def __call__(self, outputs, labels):
        total_loss = None
        if self.model == 'BRITS':
            total_loss = outputs['consistency_loss'] * self.consistency_loss_weight
        outputs['reconstruction_loss'] = outputs['reconstruction_loss'] * self.reconstruction_loss_weight
        outputs['imputation_loss'] = outputs['imputation_loss'] * self.imputation_loss_weight
        if self.MIT:
            if total_loss is not None:
                total_loss += outputs['imputation_loss']
            else:
                total_loss = outputs['imputation_loss']
        if self.ORT:
            if total_loss is not None:
                total_loss += outputs['reconstruction_loss']  
            else:
                total_loss = outputs['reconstruction_loss']  
        return total_loss

class ExampleLoss:
    def __init__(self) -> None:
        self.mse = nn.MSELoss()
        self.bce = nn.CrossEntropyLoss()

    def __call__(self, outputs, labels):
        mse_labels = labels["mean"]
        pred_labels = labels["dominant_morphology"]

        mse_outputs = outputs[0]
        pred_outputs = outputs[1]

        mse_loss = self.mse(mse_outputs, mse_labels)
        bce_loss = self.bce(pred_outputs, pred_labels)

        return mse_loss + bce_loss

class VAELoss:
    def __init__(self, C) -> None:
        self.KL = nn.KLDivLoss()
        self.MSE = nn.MSELoss()
        self.C = C
    
    def __call__(self, outputs, labels):
        reconstruction = outputs[0]
        mu = outputs[1]
        logvar = outputs[2]

        recons_loss =F.mse_loss(reconstruction, labels)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        loss = recons_loss + self.C * kld_loss
        return loss

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction="sum").div(batch_size)
    elif distribution == 'gaussian':
        # x_recon = torch.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, reduction="sum").div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    total_kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
    return total_kld

class BetaVAELoss_H:
    def __init__(self, beta, distribution="gaussian") -> None:
        self.beta = beta
        self.distribution = distribution
    
    def __call__(self, outputs, labels):
        reconstruction = outputs[0]
        mu = outputs[1]
        logvar = outputs[2]

        recon_loss = reconstruction_loss(labels, reconstruction, self.distribution)
        total_kld = kl_divergence(mu, logvar)
        return recon_loss + self.beta * total_kld

class BetaVAELoss_B:
    def __init__(self, beta, distribution="gaussian", gamma=1000, C_max=25,
    C_stop_iter=1e5) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.distribution = distribution
        self.gamma = gamma
        self.C_max = torch.autograd.Variable(torch.FloatTensor([C_max]).to(device))
        self.C_stop_iter = C_stop_iter
        self.global_iter = 0
    
    def __call__(self, outputs, labels):
        reconstruction = outputs[0]
        mu = outputs[1]
        logvar = outputs[2]

        C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
        self.global_iter += 1
        
        recon_loss = reconstruction_loss(reconstruction, labels, self.distribution)
        total_kld= kl_divergence(mu, logvar)
        beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()
        return beta_vae_loss


AVAILABLE_CRITERIONS = {
    "BCELoss": nn.BCELoss,
    "BalancedBCELoss": BalancedBCELoss,
    "CELoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
    "VAELoss": VAELoss,
    "JointForecastingLoss": ExampleLoss,
    "BetaVAELoss_H": BetaVAELoss_H,
    "BetaVAELoss_B": BetaVAELoss_B,
    "ImputationLoss": InputationLoss
}


def get_criterion(config:dict):
    if config["criterion_params"] is None:
        return AVAILABLE_CRITERIONS[config["criterion"]]()
    else:
        return AVAILABLE_CRITERIONS[config["criterion"]](**config["criterion_params"])