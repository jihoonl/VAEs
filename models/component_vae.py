import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, kl_divergence

from .vae import BaseEncoder, BaseDecoder
from .sbd import SpatialBroadcastDecoder


class ComponentEncoder(nn.Module):

    def __init__(self, d, h, w, zdim, hdim, *args, **kwargs):
        super().__init__()
        self.zdim = zdim

        self.conv1 = nn.Conv2d(d, zdim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(zdim, zdim, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(zdim,
                               zdim * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.conv4 = nn.Conv2d(zdim * 2,
                               zdim * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.mlp1 = nn.Linear(zdim * 2 * (h // 16) * (w // 16), hdim)
        self.mlp2 = nn.Linear(hdim, zdim * 2)

    def forward(self, x):
        b, *xdims = x.shape
        out = F.elu(self.conv1(x))
        out = F.elu(self.conv2(out))
        out = F.elu(self.conv3(out))
        out = F.elu(self.conv4(out))
        out = out.view(out.shape[0], -1)
        out = F.elu(self.mlp1(out))
        out = self.mlp2(out)
        return out.view(b, self.zdim * 2, 1, 1)


class ComponentVAE(nn.Module):
    """
    Component VAE specified in

    MONet: Unsupervised Scene Decomposition and Representation
    - https://arxiv.org/pdf/1901.11390.pdf
    """

    def __init__(self, d, h, w, zdim=64, hdim=128, *args, **kwargs):
        super().__init__()
        mask_dim = 1
        #self.posterior_encoder = BaseEncoder(d + mask_dim, h, w, zdim, hdim,
        #                                     *args, **kwargs)
        #self.prior_encoder = BaseEncoder(mask_dim, h, w, zdim, hdim, *args,
        #                                 **kwargs)
        self.posterior_encoder = ComponentEncoder(d + mask_dim, h, w, zdim,
                                                  hdim, *args, **kwargs)
        self.prior_encoder = ComponentEncoder(mask_dim, h, w, zdim, hdim, *args,
                                              **kwargs)

        #self.decoder = BaseDecoder(d, h, w, zdim, hdim, *args, **kwargs)
        self.decoder = SpatialBroadcastDecoder(d, h, w, zdim, hdim, *args,
                                               **kwargs)

    def forward(self, x, masks_log):
        K = len(masks_log)
        x = x.repeat(K, 1, 1, 1)
        mlog = torch.cat(masks_log, dim=0)

        # Posterior Encode
        q_encoded = self.posterior_encoder(torch.cat([x, mlog], dim=1))
        q_mu, q_logvar = torch.chunk(q_encoded, 2, dim=1)
        # q = Normal(q_mu, F.softplus(q_logvar + 0.5) + 1e-8)
        q = Normal(q_mu, q_logvar.sigmoid() * 0.99 + 0.01)

        # Prior Encode
        p_encoded = self.prior_encoder(mlog)
        p_mu, p_logvar = torch.chunk(p_encoded, 2, dim=1)
        p = Normal(p_mu, p_logvar.sigmoid() * 0.99 + 0.01)
        # p = Normal(p_mu, F.softplus(p_logvar + 0.5) + 1e-8)

        # KL(q(z^c|x,z^m)|| p(z^c|z^m))
        kl_all = kl_divergence(q, p)
        kl = torch.chunk(kl_all, K, dim=0)
        kl_k = torch.stack(torch.chunk(kl_all, K, dim=0), dim=1)

        # Decode
        z = q.rsample()
        x_mu = self.decoder(z)
        x_mu_k = torch.stack(torch.chunk(x_mu, K, dim=0), dim=1)

        return x_mu_k, kl_k
