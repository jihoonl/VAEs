import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


class VAE(nn.Module):

    def __init__(self, d, w, h, zdim=20, hdim=400, *args, **kwargs):
        super(VAE, self).__init__()

        self.stem = nn.Linear(d * w * h, hdim)
        self.gaussian = nn.Linear(hdim, zdim * 2)

        self.decoder1 = nn.Linear(zdim, hdim)
        self.decoder2 = nn.Linear(hdim, d * w * h)

    def forward(self, x):
        b, *xdims = x.shape
        mu, logvar = self.encode(x.view(b, -1))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon.view(x.shape), kl

    def encode(self, x):
        out = self.stem(x)
        out = F.elu(out)
        mu, logvar = torch.chunk(self.gaussian(out), 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        q = Normal(mu, std)
        return q.rsample()

    def decode(self, z):
        out = self.decoder1(z)
        out = F.relu(out)
        out = self.decoder2(out)
        return torch.sigmoid(out)
