import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from .conv_lstm import Conv2dLSTMCell

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')


class ConvDraw(nn.Module):
    """
    Towards Conceptual Compression
    - https://arxiv.org/pdf/1604.08772.pdf
    """

    def __init__(self,
                 xdim,
                 height,
                 width,
                 hdim,
                 zdim,
                 read_size=5,
                 write_size=5,
                 glimpse=10,
                 *args,
                 **kwargs):
        super(ConvDraw, self).__init__()
        self.xdim = xdim
        self.hdim = hdim
        self.zdim = zdim

        self.T = glimpse

        self.encoder = Conv2dLSTMCell(xdim + xdim + hdim, hdim, read_size)
        self.decoder = Conv2dLSTMCell(zdim + xdim, hdim, write_size)

        self.prior = nn.Conv2d(hdim,
                               zdim * 2,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.posterior = nn.Conv2d(hdim,
                                   zdim * 2,
                                   kernel_size=5,
                                   stride=1,
                                   padding=2)

        self.upsampler = nn.Conv2d(hdim, xdim, kernel_size=1, stride=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if hasattr(m.weight, 'bias') and m.weight.bias is not None:
                    nn.init.constant_(m.weight.bias, 0)

    def forward(self, x):
        hdim, T = self.hdim, self.T
        batch, d, h, w = x.shape

        canvas = x.new_zeros((batch, d, h, w))

        c_enc = x.new_zeros((batch, hdim, h, w))
        h_enc = x.new_zeros((batch, hdim, h, w))

        c_dec = x.new_zeros((batch, hdim, h, w))
        h_dec = x.new_zeros((batch, hdim, h, w))

        kl = 0
        for t in range(T):
            eps = x - torch.sigmoid(canvas)

            c_enc, h_enc = self.encoder(torch.cat([x, eps, h_dec], dim=1),
                                        (c_enc, h_enc))

            # Prior
            p_mu, p_logvar = torch.chunk(self.prior(h_dec), 2, dim=1)
            p = Normal(p_mu, p_logvar.mul(0.5).exp())

            # Posterior
            q_mu, q_logvar = torch.chunk(self.posterior(h_enc), 2, dim=1)
            q = Normal(q_mu, q_logvar.mul(0.5).exp())

            # Sample
            z = q.rsample()

            c_dec, h_dec = self.decoder(torch.cat([z, canvas], dim=1),
                                        (c_dec, h_dec))

            current = self.upsampler(h_dec)
            canvas = canvas + current
            kl += kl_divergence(q, p)

        kl = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

        return torch.sigmoid(canvas.view(x.shape)), kl
