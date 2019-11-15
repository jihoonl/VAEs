import torch
import torch.nn.functional as F
from torch import nn

from .sbp import LatentSBP
from .component_vae import ComponentVAE


class Genesis(nn.Module):

    def __init__(self, d, h, w, zdim=64, hdim=128, layers=4, *args, **kwargs):
        self.layers = layers
        self.zdim = zdim
        self.hdim = hdim

        self.mask_vae = LatentSBP(d, h, w, zdim, hdim, *args, **kwargs)

        self.component_vae = ComponentVAE(d, h, w, zdim, hdim, *args, **kwargs)

    def forward(self, x):
        """
        1. Mask encoding by Stick Breaking Process Encoder
        2. Component VAE to encode and decode.
        """
        layers = self.layers

        # Stick Breaking Process
        masks, kl_mask = self.mask_encoder(x, layers)

        x_mu, kl = self.component_vae(x, masks)

        return x_mu, masks, kl
