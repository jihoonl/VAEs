import torch
import torch.nn.functional as F
from torch import nn

from .vae import BaseEncoder, BaseDecoder


class ComponentVAE(nn.Module):

    def __init__(self, d, h, w, zdim=64, hdim=128, *args, **kwargs):
        self.posterior_encoder = BaseEncoder(d, h, w, zdim, hdim, *args,
                                             **kwargs)
        self.posterior_decoder = BaseDecoder(d, h, w, zdim, hdim, *args,
                                             **kwargs)

        self.prior_encoder = BaseEncoder(d, h, w, zdim, hdim, *args, **kwargs)

    def forward(self, x, masks):
        K = len(masks)
        x = x.repeat(K, 1, 1, 1)
        m = torch.cat(masks, dim=0)

        encoded = self.posterior_encoder(torch.cat([x, m], dim=1))
