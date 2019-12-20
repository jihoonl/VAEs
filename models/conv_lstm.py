import torch
from torch import nn


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

    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_size=5,
                 stride=1,
                 padding=1,
                 use_bn=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.forget = nn.Sequential(
            nn.Conv2d(in_channels=in_dim + out_dim,
                      out_channels=out_dim,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=not use_bn),
            nn.BatchNorm2d(num_features=out_dim) if use_bn else nn.Identity(),
            nn.Sigmoid())
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=in_dim + out_dim,
                      out_channels=out_dim,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=not use_bn),
            nn.BatchNorm2d(num_features=out_dim) if use_bn else nn.Identity(),
            nn.Sigmoid())
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=in_dim + out_dim,
                      out_channels=out_dim,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=not use_bn),
            nn.BatchNorm2d(num_features=out_dim) if use_bn else nn.Identity(),
            nn.Sigmoid())

        self.state = nn.Sequential(
            nn.Conv2d(in_channels=in_dim + out_dim,
                      out_channels=out_dim,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=not use_bn),
            nn.BatchNorm2d(num_features=out_dim) if use_bn else nn.Identity(),
            nn.Tanh())

    def forward(self, input_, states):
        cell, hidden = states

        i = torch.cat((hidden, input_), dim=1)
        forget_gate = self.forget(i)
        input_gate = self.input(i)
        output_gate = self.output(i)
        state_gate = self.state(i)

        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)
        return cell, hidden
