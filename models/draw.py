import torch
from torch import nn
from torch.distributions import Normal

from .convlstm import Conv2dLSTMCell


class Draw(nn.Module):
    """
    DRAW: A Recurrent Neural Network For Image Generation
    - http://arxiv.org/pdf/1502.04623v2.pdf
    Implementation based on Eric Jang's code
    """

    def __init__(self, xdim, height, width, hdim, zdim, T=10, attention=False):
        super(Draw, self).__init__()
        self.xdim = xdim
        self.hdim = hdim
        self.zdim = zdim
        self.read = self.read_attention if attention else self.read_no_attention

        self.write = self.write_attention if attention else nn.Linear(
            hdim, xdim * height * width)
        self.T = T

        self.sampler = nn.Linear(hdim, zdim * 2)
        self.encoder = nn.LSTMCell(xdim * height * width * 2 + hdim, hdim)
        self.decoder = nn.LSTMCell(zdim, hdim)

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
        x_flatten = x.view(batch, -1)

        canvas = x.new_zeros((batch, d * h * w))

        c_enc = x.new_zeros((batch, hdim))
        h_enc = x.new_zeros((batch, hdim))

        c_dec = x.new_zeros((batch, hdim))
        h_dec = x.new_zeros((batch, hdim))

        kl = 0
        for t in range(T):
            x_hat = x_flatten - torch.sigmoid(canvas)
            r = self.read(x_flatten, x_hat, h_dec)

            # Inference Core
            h_enc, c_enc = self.encoder(torch.cat([r, h_dec], 1),
                                        (h_enc, c_enc))
            z, mu, logvar = self.sample_q(h_enc.view(batch, -1))

            # Generation Core
            h_dec, c_dec = self.decoder(z, (h_dec, c_dec))

            # Update canvas
            canvas = canvas + self.write(h_dec)
            kl += torch.sum(mu.pow(2) + logvar.exp() - logvar)

        # In DRAW paper Eq.10. KL Divergence
        # KL = {1\over{2}}(\sum^T_{t=1}\mu^2_t+\sigma^2_t-log\sigma^2_t)-T/2
        kl = 0.5 * kl - T / 2
        return torch.sigmoid(canvas.view(x.shape)), kl

    def sample_q(self, h):
        mu, logvar = torch.chunk(self.sampler(h), 2, dim=1)
        q = Normal(mu, logvar.mul(0.5).exp())
        return q.rsample(), mu, logvar

    def read_no_attention(self, x, x_hat, h_dec):
        return torch.cat([x, x_hat], 1)

    def read_attention(self, x, x_hat, h_dec):
        raise NotImplementedError()

    def write_attention(self, h_dec):
        raise NotImplementedError()
