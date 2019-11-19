import torch
import torch.nn.functional as F
from torch import nn

from .component_vae import ComponentVAE
from .sbp import RecurrentSBP, TowerRecurrentSBP


class Genesis(nn.Module):

    def __init__(self, d, h, w, zdim=64, hdim=128, layers=4, *args, **kwargs):
        super().__init__()
        self.layers = layers
        self.zdim = zdim
        self.hdim = hdim

        self.mask_vae = RecurrentSBP(d, h, w, zdim, hdim, *args, **kwargs)
        self.component_vae = ComponentVAE(d, h, w, zdim, hdim, *args, **kwargs)

    def forward(self, x):
        """
        1. Mask encoding by Stick Breaking Process Encoder
        2. Component VAE to encode and decode. image
        """
        layers = self.layers

        # pi - Stick Breaking Process
        log_ms_k, kl_m = self.mask_vae(x, layers)

        # Decode components
        x_mu_k, kl_c = self.component_vae(x, log_ms_k)
        ms_k = torch.stack(log_ms_k, dim=4).exp()

        recon_k = ms_k * x_mu_k
        recon = recon_k.sum(dim=4)

        return recon, x_mu_k, ms_k, kl_m, kl_c
