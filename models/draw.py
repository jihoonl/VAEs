import torch
from torch import nn
from torch.distributions import Normal

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')


class Draw(nn.Module):
    """
    DRAW: A Recurrent Neural Network For Image Generation
    - http://arxiv.org/pdf/1502.04623v2.pdf
    Implementation based on Eric Jang's code,
        and https://github.com/Natsu6767/Generating-Devanagari-Using-DRAW
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
                 attention=False):
        super(Draw, self).__init__()
        self.xdim = xdim
        self.hdim = hdim
        self.zdim = zdim

        if attention:
            self.read_attn_window = nn.Linear(self.hdim, 5)
            self.read_n = read_size
            self.read = self.read_attention
            self.write_n = write_size
            self.write_attn_window = nn.Linear(self.hdim, 5)
            self.write = self.write_attention
            self.encoder = nn.LSTMCell(
                xdim * self.read_n * self.read_n * 2 + hdim, hdim)
            self.write_fc = nn.Linear(hdim, xdim * self.write_n * self.write_n)
        else:
            self.read = self.read_no_attention
            self.write = self.write_no_attention
            self.encoder = nn.LSTMCell(xdim * height * width * 2 + hdim, hdim)
            self.write_fc = nn.Linear(hdim, xdim * height * width)

        self.T = glimpse

        self.sampler = nn.Linear(hdim, zdim * 2)
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
        # x_flatten = x.view(batch, -1)

        canvas = x.new_zeros((batch, d, h, w))

        c_enc = x.new_zeros((batch, hdim))
        h_enc = x.new_zeros((batch, hdim))

        c_dec = x.new_zeros((batch, hdim))
        h_dec = x.new_zeros((batch, hdim))

        kl = 0
        for t in range(T):
            x_hat = x - torch.sigmoid(canvas)
            r = self.read(x, x_hat, h_dec, x.shape)

            # Inference Core
            h_enc, c_enc = self.encoder(torch.cat([r, h_dec], 1),
                                        (h_enc, c_enc))
            z, mu, logvar = self.sample_q(h_enc.view(batch, -1))

            # Generation Core
            h_dec, c_dec = self.decoder(z, (h_dec, c_dec))

            # Update canvas
            canvas = canvas + self.write(h_dec, x.shape)
            kl += torch.sum(mu.pow(2) + logvar.exp() - logvar)

        # In DRAW paper Eq.10. KL Divergence
        # KL = {1\over{2}}(\sum^T_{t=1}\mu^2_t+\sigma^2_t-log\sigma^2_t)-T/2
        kl = 0.5 * kl - T / 2
        return torch.sigmoid(canvas.view(x.shape)), kl

    def sample_q(self, h):
        mu, logvar = torch.chunk(self.sampler(h), 2, dim=1)
        q = Normal(mu, logvar.mul(0.5).exp())
        return q.rsample(), mu, logvar

    def read_no_attention(self, x, x_hat, h_dec, shape):
        batch, *xdim = x.shape
        return torch.cat([x.view(batch, -1), x_hat.view(batch, -1)], 1)

    def write_no_attention(self, h_dec, shape):
        return self.write_fc(h_dec).view(shape)

    def read_attention(self, x, x_hat, h_dec, shape):
        batch, d, h, w = shape
        (Fx, Fy), gamma = self.attn_window(h_dec, self.read_n,
                                           self.read_attn_window, w, h)

        Fx = Fx.repeat(1, d, 1, 1)
        Fy = Fy.repeat(1, d, 1, 1)

        def filter_img(img, Fx, Fy, gamma, N):
            xFx = torch.matmul(img, Fx.permute(0, 1, 3, 2))
            glimpse = torch.matmul(Fy, xFx)
            glimpse = glimpse.view(-1, d * self.read_n * self.read_n)
            return glimpse * gamma.view(batch, -1).expand_as(glimpse)

        x = filter_img(x, Fx, Fy, gamma, self.read_n)
        x_hat = filter_img(x_hat.view(batch, d, h, w), Fx, Fy, gamma,
                           self.read_n)
        return torch.cat([x.view(batch, -1), x_hat.view(batch, -1)], 1)

    def write_attention(self, h_dec, shape):
        batch, d, h, w = shape

        Wt = self.write_fc(h_dec).view(batch, d, self.write_n, self.write_n)
        (Fx, Fy), gamma = self.attn_window(h_dec, self.write_n,
                                           self.write_attn_window, w, h)

        Fyt = Fy.permute(0, 1, 3, 2)
        WtFx = torch.matmul(Wt, Fx)
        wr = torch.matmul(Fyt, WtFx)
        wr = wr.view(batch, -1)
        return (wr / gamma.view(batch, -1).expand_as(wr)).view(shape)

    def attn_window(self, h_dec, N, W, A, B):
        # (\tilde{g}_X, \tilde{g}_Y, log\sigma^2, log\tilde\delta, log\gamma = \
        #       W(h^{dec})
        # g_X={A+1\over2}(\tilde{g}_X+1)
        # g_Y={B+1\over2}(\tilde{g}_Y+1)
        # \delta={max(A,B)-1\over{N-1}}\tilde\delta
        param = W(h_dec)
        gx_, gy_, logvar, logdelta, loggamma = torch.chunk(param, 5, dim=1)
        gx = (A + 1) / 2 * (gx_ + 1)
        gy = (B + 1) / 2 * (gy_ + 1)
        var = logvar.exp()
        delta = (max(A, B) - 1) / (N - 1) * logdelta.exp()
        return self.filterbank(gx, gy, var, delta, N, A, B), loggamma.exp()

    def filterbank(self, gx, gy, var, delta, N, A, B, eps=1e-8):

        # \mu^i_X=g_X+(i-N/2 - 0.5)\delta
        # \mu^i_Y=g_Y+(j-N/2 - 0.5)\delta
        grid_i = torch.arange(end=N).float().to(device).view(1, -1)

        mu_x = gx + (grid_i - N / 2 - 0.5) * delta
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta
        mu_x = mu_x.view(-1, N, 1)
        mu_y = mu_y.view(-1, N, 1)
        var = var.view(-1, 1, 1)

        a = torch.arange(end=A).float().to(device).view(1, 1, -1)
        b = torch.arange(end=B).float().to(device).view(1, 1, -1)

        # F_X[i,a] = {1\over Z_X}exp\Big(-{(a-\mu^i_X)^2 \over 2\sigma^2}\Big)
        # F_X[j,b] = {1\over Z_Y}exp\Big(-{(a-\mu^j_Y)^2 \over 2\sigma^2}\Big)
        Fx = torch.exp(-torch.pow(a - mu_x, 2) / (2 * var))
        Fy = torch.exp(-torch.pow(b - mu_y, 2) / (2 * var))

        Fx = Fx / (Fx.sum(dim=2, keepdim=True).expand_as(Fx) + eps)
        Fy = Fy / (Fy.sum(dim=2, keepdim=True).expand_as(Fy) + eps)
        return Fx.unsqueeze(1), Fy.unsqueeze(1)
