import torch
import torch.nn.functional as F
from torch import nn


class VAE(nn.Module):

    def __init__(self, d, w, h):
        super(VAE, self).__init__()

        self.stem = nn.Linear(w * h, 400)
        self.encode_mu = nn.Linear(400, 20)
        self.encode_logvar = nn.Linear(400, 20)

        self.decoder1 = nn.Linear(20, 400)
        self.decoder2 = nn.Linear(400, w * h)

    def forward(self, x):
        b, *xdims = x.shape
        mu, logvar = self.encode(x.view(b, -1))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon.view(x.shape), mu, logvar

    def encode(self, x):
        out = self.stem(x)
        out = F.relu(out)
        return self.encode_mu(out), self.encode_logvar(out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        out = self.decoder1(z)
        out = F.relu(out)
        out = self.decoder2(out)
        return torch.sigmoid(out)
