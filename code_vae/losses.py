# losses.py
import torch
import torch.nn.functional as F

def binary_cross_entropy_loss(recon_x, x):
    """
    Element-wise BCE (sum over features).
    recon_x is assumed in (0,1) when sigmoid is used.
    """
    return -torch.sum(
        x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8),
        dim=-1
    )

def compute_kl_divergence(mu, logvar):
    """KL(q(z|x) || p(z)) for diagonal Gaussian."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def compute_elbo_loss(recon_x, x, z_params, binary=True):
    """
    Return (sum_log_likelihood, sum_kl) over the batch.
    """
    mu, logvar = z_params
    kld = compute_kl_divergence(mu, logvar)

    if binary:
        likelihood = -binary_cross_entropy_loss(recon_x, x)
    else:
        # MSE per sample, summed over features
        likelihood = -F.mse_loss(recon_x, x, reduction="none").sum(dim=1)

    return torch.sum(likelihood), torch.sum(kld)
