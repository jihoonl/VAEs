import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, kl_divergence

from .vae import BaseEncoder, BaseDecoder


class ComponentVAE(nn.Module):
    """
    Component VAE specified in

    MONet: Unsupervised Scene Decomposition and Representation
    - https://arxiv.org/pdf/1901.11390.pdf
    """

    def __init__(self, d, h, w, zdim=64, hdim=128, *args, **kwargs):
        super().__init__()
        mask_dim = 1
        self.posterior_encoder = BaseEncoder(d + mask_dim, h, w, zdim, hdim,
                                             *args, **kwargs)
        self.posterior_decoder = BaseDecoder(d, h, w, zdim, hdim, *args,
                                             **kwargs)

        self.prior_encoder = BaseEncoder(mask_dim, h, w, zdim, hdim, *args,
                                         **kwargs)

    def forward(self, x, masks_log):
        K = len(masks_log)
        x = x.repeat(K, 1, 1, 1)
        mlog = torch.cat(masks_log, dim=0)

        # Posterior Encode
        q_encoded = self.posterior_encoder(torch.cat([x, mlog], dim=1))
        q_mu, q_sigma = torch.chunk(q_encoded, 2, dim=1)
        q = Normal(q_mu, q_sigma)

        # Prior Encode
        p_encoded = self.prior_encoder(mlog)
        p_mu, p_sigma = torch.chunk(p_encoded, 2, dim=1)
        p = Normal(p_mu, p_sigma)
        kl_all = kl_divergence(q, p)
        kl_k = torch.stack(torch.chunk(kl_all, K, dim=0), dim=4)

        # Decode
        z = q.rsample()
        x_mu = self.posterior_decoder(z)
        x_mu_k = torch.stack(torch.chunk(x_mu, K, dim=0), dim=4)

        return x_mu_k, kl_k
