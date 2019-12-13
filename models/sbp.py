import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, kl_divergence

from .vae import BaseEncoder, BaseDecoder, VAE, TowerVAE
from .conv_draw import Conv2dLSTMCell
from .timer import Timer
from .modules.sylvester_vae import VAE as SylvesterVAE


def reparameterize(mu, logstd):
    q = Normal(mu, logstd.sigmoid() * 0.99 + 0.01)
    # q = Normal(mu, F.softplus(logstd + 0.5) + 1e-8)
    z = q.rsample()
    return q, z


class TowerRecurrentSBP(nn.Module):

    def __init__(self, d, h, w, zdim, hdim, *args, **kwargs):
        super().__init__()
        self.core = TowerVAE(d,
                             h,
                             w,
                             zdim,
                             hdim,
                             odim=1,
                             batch_norm=True,
                             *args,
                             **kwargs)

        #self.posterior_lstm = nn.LSTM(zdim + hdim, zdim * 2)
        #self.posterior_linear = nn.Linear(zdim * 2, zdim * 2)
        self.posterior_lstm = Conv2dLSTMCell(hdim + hdim,
                                             hdim,
                                             kernel_size=5,
                                             stride=1,
                                             padding=2)
        self.posterior_linear = nn.Conv2d(hdim,
                                          zdim * 2,
                                          kernel_size=1,
                                          stride=1)

        self.prior_lstm = Conv2dLSTMCell(zdim,
                                         hdim,
                                         kernel_size=5,
                                         stride=1,
                                         padding=2)
        self.prior_linear = nn.Conv2d(hdim, zdim * 2, kernel_size=1, stride=1)

    def forward(self, x, K):
        batch, *xdims = x.shape
        kl = 0

        t = Timer()

        # Posterior
        t.tic()
        h = self.core.encoder.stem(x)
        encoded = self.core.encoder.gaussian(h)
        mu, logvar = torch.chunk(encoded, 2, dim=1)
        q, z = reparameterize(mu, logvar)
        zs_q = [z]

        # Prior
        p = Normal(0, 1)

        # First layer kl divergence
        kl = []
        kl.append(kl_divergence(q, p))

        # z^m encoding step
        batch, zdim, *shape = z.shape
        c_q = x.new_zeros((batch, zdim * 2, *shape))
        h_q = x.new_zeros((batch, zdim * 2, *shape))

        c_p = x.new_zeros((batch, zdim * 2, *shape))
        h_p = x.new_zeros((batch, zdim * 2, *shape))
        """
        Genesis Eq - 1
        p_{\theta}(z^m_{1:K}) =
            \prod^K_{k=1}p_{\theta}(z^m_k|u_k)|_u_k=R_\theta(z^m_{k-1},u_{k-1})
        """
        t.tic()
        for s in range(K - 1):
            # Posterior
            c_q, h_q = self.posterior_lstm(torch.cat([h, zs_q[-1]], dim=1),
                                           (c_q, h_q))
            q_encoded = self.posterior_linear(h_q)
            q_mu, q_logvar = torch.chunk(q_encoded, 2, dim=1)
            q, z_q = reparameterize(q_mu, q_logvar)

            # Prior
            c_p, h_p = self.prior_lstm(zs_q[-1], (c_p, h_p))
            p_encoded = self.prior_linear(h_p)
            p_mu, p_logvar = torch.chunk(p_encoded, 2, dim=1)
            p, z_p = reparameterize(p_mu, p_logvar)

            zs_q.append(z_q)
            kl.append(kl_divergence(q, p))
        t.tic()
        kl = torch.stack(kl, dim=4)

        # Parallelized decoding of z^m
        zs_q = torch.cat(zs_q, dim=0)
        decoded_zs = self.core.decoder(zs_q)
        decoded_zs = torch.chunk(decoded_zs, K, dim=0)

        t.tic()
        # Genesis Eq - 4 log version
        log_ms = []
        log_ss = [torch.zeros_like(x)[:, :1, :, :]]
        # pi_1:K-1
        for z in decoded_zs[:-1]:
            log_m = F.logsigmoid(z)
            log_neg_m = F.logsigmoid(-z)
            log_ms.append(log_ss[-1] + log_m)
            log_ss.append(log_ss[-1] + log_neg_m)
        # pi_K
        log_neg_m = log_ss[-1] + F.logsigmoid(decoded_zs[-1])
        log_ms.append(log_neg_m)
        return log_ms, kl


class GatedRecurrentSBP(nn.Module):

    def __init__(self, d, h, w, zdim, hdim, *args, **kwargs):
        super().__init__()
        self.core = SylvesterVAE(z_size=zdim,
                                 input_size=(d, h, w),
                                 nout=1,
                                 h_size=hdim,
                                 enc_norm='bn',
                                 dec_norm='bn')

        self.posterior_lstm = nn.LSTM(hdim + hdim, hdim)
        self.posterior_linear = nn.Linear(hdim, zdim * 2)

        self.prior_lstm = nn.LSTM(zdim, hdim)
        self.prior_linear = nn.Linear(hdim, zdim * 2)

        self.zdim = zdim
        self.hdim = hdim

    def forward(self, x, K):
        batch, *xdims = x.shape

        # Posterior
        h = self.core.q_z_nn(x).view(batch, -1)
        h = h.view(1, batch, -1)

        # z^m encoding step
        """
        Genesis Eq - 1
        p_{\theta}(z^m_{1:K}) =
            \prod^K_{k=1}p_{\theta}(z^m_k|u_k)|_u_k=R_\theta(z^m_{k-1},u_{k-1})
        """
        q_state = (x.new_zeros([1, batch,
                                self.hdim]), x.new_zeros([1, batch, self.hdim]))
        p_state = (x.new_zeros([1, batch,
                                self.hdim]), x.new_zeros([1, batch, self.hdim]))
        zs_q = []
        kl = []
        for s in range(K):
            # Prior dist
            p_encoded = self.prior_linear(p_state[0].view(batch, -1))
            p_mu, p_logstd = torch.chunk(p_encoded, 2, dim=1)
            p, z_p = reparameterize(p_mu, p_logstd)

            # Posterior rnn
            _, q_state = self.posterior_lstm(torch.cat([h, p_state[0]], dim=2),
                                             q_state)

            # posterior dist
            q_encoded = self.posterior_linear(q_state[0].view(batch, -1))
            q_mu, q_logstd = torch.chunk(q_encoded, 2, dim=1)

            # Posterior sample
            q, z_q = reparameterize(q_mu, q_logstd)
            zs_q.append(z_q)

            # Prior rnn
            _, p_state = self.prior_lstm(z_q.view(1, batch, -1), p_state)

            # Update kl divergence
            kl.append(kl_divergence(q, p).view(batch, -1, 1, 1))

        kl = torch.stack(kl, dim=4)

        # Parallelized decoding of z^m
        zs_q = torch.cat(zs_q, dim=0)
        decoded_zs = self.core.decode(zs_q)
        decoded_zs = torch.chunk(decoded_zs, K, dim=0)

        # Genesis Eq - 4 log version
        log_ms = []
        log_ss = [torch.zeros_like(x)[:, :1, :, :]]
        # pi_1:K-1
        for z in decoded_zs[:-1]:
            log_m = F.logsigmoid(z)
            log_neg_m = F.logsigmoid(-z)
            log_ms.append(log_ss[-1] + log_m)
            log_ss.append(log_ss[-1] + log_neg_m)
        # pi_K
        log_neg_m = log_ss[-1]
        log_ms.append(log_neg_m)
        return log_ms, kl
