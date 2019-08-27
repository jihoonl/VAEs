import torch
from torch import nn
from .convlstm import Conv2dLSTMCell


class Draw(nn.Module):
    """
    DRAW: A Recurrent Neural Network For Image Generation - http://arxiv.org/pdf/1502.04623v2.pdf
    Implementation based on Eric Jang's code
    """

    def __init__(self, xdim, hdim, zdim, L=10, share=True):
        self.xdim = xdim
        self.hdim = hdim
        self.zdim = zdim
        self.share = share
        self.L = L

        self.sampler = nn.Linear(hdim, zdim * 2)
        self.encoder = Conv2dLSTMCell(xdim, hdim)
        self.decoder = Conv2dLSTMCell(xdim, hdim)

    def forward(self, x):
        hdim, L = self.hdim, self.L
        batch, d, h, w = x.shape

        canvas = x.new_zeros((batch, d * h * w))

        c_dec = x.new_zeros((batch, hdim))
        h_dec = x.new_zeros((batch, hdim))

        c_enc = x.new_zeros((batch, hdim))
        h_enc = x.new_zeros((batch, hdim))

        for l in range(L):
            x_hat = x - torch.sigmoid(canvas)
            r = self.read(x, x_hat, h_dec)
            h_enc, c_enc = self.encode(c_enc, r, h_dec)

    def read(self, x, x_hat, h_dec):
        raise NotImplementedError()

    def encode(self, c_enc, r, h_dec):
        raise NotImplementedError()
