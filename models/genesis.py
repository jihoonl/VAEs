import torch
import torch.nn.functional as F
from torch import nn

from .component_vae import ComponentVAE
from .sbp import RecurrentSBP, TowerRecurrentSBP, GatedRecurrentSBP
from .timer import Timer


class Genesis(nn.Module):

    def __init__(self, d, h, w, zdim=64, hdim=128, layers=4, *args, **kwargs):
        super().__init__()
        self.layers = layers
        self.zdim = zdim
        self.hdim = hdim

        # self.mask_vae = TowerRecurrentSBP(d, h, w, zdim, hdim, *args, **kwargs)
        self.mask_vae = GatedRecurrentSBP(d, h, w, zdim, hdim, *args, **kwargs)
        self.component_vae = ComponentVAE(d, h, w, zdim, hdim, *args, **kwargs)

    def forward(self, x):
        """
        1. Mask encoding by Stick Breaking Process Encoder
        2. Component VAE to encode and decode. image
        """
        layers = self.layers
        t = Timer()

        # pi - Stick Breaking Process
        t.tic()
        log_ms_k, kl_m = self.mask_vae(x, layers)
        # print('Mask VAE: ', t.toc())

        # Decode components
        t.tic()
        x_mu_k, kl_c = self.component_vae(x, log_ms_k)
        #print('Comp VAE: ', t.toc())
        t.tic()
        log_ms_k = torch.stack(log_ms_k, dim=4)

        recon_k = log_ms_k.exp() * x_mu_k
        recon = recon_k.sum(dim=4)
        #print('Remain VAE: ', t.toc())

        return recon, recon_k, x_mu_k, log_ms_k, kl_m, kl_c
