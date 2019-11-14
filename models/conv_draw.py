import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')


class Conv2dLSTMCell(nn.Module):
    """
    Convolutional LSTM - http://arxiv.org/abs/1506.04214
    with conventional LSTM implementation which omits peephole connection

    input_gate i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i)
    forget_gate f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f)
    output_gate o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o)
    state_gate s_t = tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)

    Next Cell c_t = f_t \circ c_{t-1} + i_t \circ s_t
    Next Hidden h_t = o_t \circ tanh(c_t)
    """

    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=2):
        super(Conv2dLSTMCell, self).__init__()
        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        # Computes input, forget, output, state all in once
        self.combined_conv = nn.Conv2d(in_dim + out_dim, out_dim * 4, **kwargs)

    def forward(self, input_, state):
        cell, hidden = state

        i = torch.cat((hidden, input_), dim=1)
        combined = self.combined_conv(i)
        c_forget, c_input, c_output, c_state = torch.chunk(combined, 4, dim=1)
        forget_gate = torch.sigmoid(c_forget)
        input_gate = torch.sigmoid(c_input)
        output_gate = torch.sigmoid(c_output)
        state_gate = torch.tanh(c_state)

        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)
        return cell, hidden


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
