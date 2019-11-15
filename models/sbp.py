import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, kl_divergence

from .vae import VAE
from .conv_draw import Conv2dLSTMCell


class RecurrentSBP(nn.Module):

    def __init__(self, d, h, w, zdim, hdim, *args, **kwargs):
        super().__init__()
        self.core = VAE(d, h, w, zdim, hdim, *args, **kwargs)

        self.prior_lstm = nn.LSTM(zdim, zdim * 2)
        self.prior_linear = nn.Linear(zdim * 2, zdim * 2)

        self.posterior_lstm = nn.LSTM(zdim, zdim * 2)
        self.posterior_linear = nn.Linear(zdim * 2, zdim * 2)

    def forward(self, x, steps):
        batch, *xdims = x.shape
        kl = 0

        # Posterior
        h = self.core.encoder.stem(x)
        encoded = self.core.encoder.gaussian(h)
        mu, logvar = torch.chunk(encoded, 2, dim=1)
        q, z = self.reparameterize(mu, logvar)
        zs_q = [z]

        # Prior
        p = Normal(0, 1)

        # First layer kl divergence
        kl += kl_divergence(q, p)

        # z^m encoding step
        q_state = None
        p_state = None
        for s in range(steps):
            # Posterior
            q_lstm, state = self.posterior_lstm(torch.cat([h, zs_q[-1]], dim=1),
                                                q_state)
            q_encoded = self.posterior_linear(q_lstm)
            q_mu, q_logvar = torch.chunk(q_encoded, 2, dim=1)
            q, z_q = self.reparameterize(mu, logvar)

            # Prior
            p_lstm, state = self.prior_lstm(zs[-1], p_state)
            p_encoded = self.prior_linear(p_lstm)
            p_mu, p_logvar = torch.chunk(p_encoded, 2, dim=1)
            p, z_p = self.reparameterize(mu, logvar)

            zs_q.append(z_q)
            kl += kl_divergence(q, p)

        # Decoding z to create mask image
        zs_q = torch.cat(zs_q, dim=0)
        masks = self.core.decoder(zs_q)
        masks = torch.chunk(mask, steps, dim=0)
        masks = masks[:, :1, :, :]

        log_ms = []
        log_ss = [torch.zeros_like(x)[:, :1, :, :]]
        for m in masks:
            log_m = F.logsigmoid(m)
            log_neg_m = F.logsigmoid(-m)
            log_ms.append(log_ss[-1] + log_m)
            log_ss.append(log_ss[-1] + log_neg_m)
        log_ms.append(log_ss[-1])
        return log_ms, kl

    def reparameterize(self, mu, logvar):
        q = Normal(mu, logvar.mul(0.5).exp())
        z = q.rsample()
        return q, z
