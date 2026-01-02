# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from losses import compute_elbo_loss

def create_mlp(layers, activation=nn.LeakyReLU(), bn=False, dropout=0.0):
    """Build an MLP (Linear + optional BN + activation + optional dropout)."""
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
        if dropout and dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

class Stochastic(nn.Module):
    """Base class for stochastic layers using reparameterization trick."""
    def reparametrize(self, mu, logvar):
        eps = torch.randn_like(mu, requires_grad=False)
        std = (0.5 * logvar).exp()
        return mu + std * eps

class GaussianSample(Stochastic):
    """Gaussian latent variable sampler producing z, mu, logvar."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.log_var(x)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

class Encoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0.0):
        """
        dims: [x_dim, hidden_dims(list), z_dim]
        """
        super().__init__()
        x_dim, h_dim, z_dim = dims
        self.hidden = create_mlp([x_dim] + h_dim, bn=bn, dropout=dropout)
        self.sample = GaussianSample(([x_dim] + h_dim)[-1], z_dim)

    def forward(self, x):
        h = self.hidden(x)
        return self.sample(h)

class Decoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0.0, output_activation=None):
        """
        dims: [z_dim, hidden_dims(list), x_dim]
        """
        super().__init__()
        z_dim, h_dim, x_dim = dims
        self.hidden = create_mlp([z_dim] + list(h_dim), bn=bn, dropout=dropout)
        self.reconstruction = nn.Linear(([z_dim] + list(h_dim))[-1], x_dim)
        self.output_activation = output_activation

    def forward(self, z):
        h = self.hidden(z)
        x_hat = self.reconstruction(h)
        if self.output_activation is not None:
            x_hat = self.output_activation(x_hat)
        return x_hat

class VAE(nn.Module):
    def __init__(self, dims, bn=False, dropout=0.0, binary=False):
        """
        dims: [x_dim, z_dim, encode_dim(list), decode_dim(list)]
        """
        super().__init__()
        x_dim, z_dim, encode_dim, decode_dim = dims
        self.binary = binary

        decode_activation = nn.Sigmoid() if binary else None
        self.encoder = Encoder([x_dim, encode_dim, z_dim], bn=bn, dropout=dropout)
        self.decoder = Decoder([z_dim, decode_dim, x_dim], bn=bn, dropout=dropout,
                               output_activation=decode_activation)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x

    def compute_loss(self, x):
        """
        Return positive losses: recon_loss, kl_loss
        recon_loss corresponds to -log p(x|z)
        """
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        likelihood, kl = compute_elbo_loss(recon_x, x, (mu, logvar), binary=self.binary)
        recon_loss = -likelihood
        kl_loss = kl
        return recon_loss, kl_loss

    @torch.no_grad()
    def encode_batch(self, dataloader, device="cpu", output_type="mu"):
        """
        Encode dataset and return:
        - 'mu': mean of q(z|x)
        - 'z': sampled z
        - 'log_var': log variance
        - 'x': reconstructed x
        """
        self.eval()
        self.to(device)
        outputs = []

        for x in dataloader:
            # dataloader yields a tensor batch directly
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.view(x.size(0), -1).float().to(device)

            z, mu, logvar = self.encoder(x)
            if output_type == "mu":
                outputs.append(mu.detach().cpu())
            elif output_type == "z":
                outputs.append(z.detach().cpu())
            elif output_type == "log_var":
                outputs.append(logvar.detach().cpu())
            elif output_type == "x":
                recon_x = self.decoder(z)
                outputs.append(recon_x.detach().cpu())
            else:
                raise ValueError(f"Unknown output_type: {output_type}")

        return torch.cat(outputs, dim=0).numpy()
