import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from .tower import TowerEncoder, TowerDecoder
from .sbd import SpatialBroadcastDecoder


class BaseEncoder(nn.Module):

    def __init__(self, d, h, w, zdim, hdim, *args, **kwargs):
        super().__init__()
        self.stem = nn.Linear(d * w * h, hdim)
        self.gaussian = nn.Linear(hdim, zdim * 2)
        self.zdim = zdim

    def forward(self, x):
        b, *xdims = x.shape

        out = F.elu(self.stem(x.view(b, -1)))
        out = self.gaussian(out)
        return out.view(b, self.zdim * 2, 1, 1)


class BaseDecoder(nn.Module):

    def __init__(self, d, h, w, zdim, hdim, *args, **kwargs):
        super().__init__()
        if 'odim' in kwargs:
            d = kwargs['odim']
        self.decoder1 = nn.Linear(zdim, hdim)
        self.decoder2 = nn.Linear(hdim, d * w * h)
        self.x_shape = (d, h, w)

    def forward(self, z):
        b, *zdims = z.shape
        out = self.decoder1(z.view(b, -1))
        out = F.elu(out)
        out = self.decoder2(out)
        return torch.sigmoid(out).view(-1, *self.x_shape)


class AbstractVAE(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        b, *xdims = x.shape
        encoded = self.encoder(x)

        mu, logvar = torch.chunk(encoded, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, kl

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        q = Normal(mu, std)
        return q.rsample()


class VAE(AbstractVAE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = BaseEncoder(*args, **kwargs)
        self.decoder = BaseDecoder(*args, **kwargs)


class SbdVAE(AbstractVAE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = BaseEncoder(*args, **kwargs)
        self.decoder = SpatialBroadcastDecoder(*args, **kwargs)


class TowerVAE(AbstractVAE):
    """
    Convolutional VAE with 8 layers
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Encoder
        self.encoder = TowerEncoder(*args, **kwargs)
        self.decoder = TowerDecoder(*args, **kwargs)


class TowerSBDVAE(AbstractVAE):

    def __init__(self, d, h, w, zdim=64, hdim=128, *args, **kwargs):
        super().__init__()

        # Encoder
        self.encoder = TowerEncoder(d, h, w, zdim, hdim, *args, **kwargs)
        self.decoder = SpatialBroadcastDecoder(d, h, w, zdim, hdim, *args,
                                               **kwargs)
