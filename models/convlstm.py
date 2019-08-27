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

    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()
        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        # Computes input, forget, output, state all in once
        self.combined_conv = nn.Conv2d(in_dim + out_dim, out_dim * 4, **kwargs)

    def forward(self, input_, state):
        cell, hidden = state

        i = torch.cat((hidden, input_), dim=1)
        combined = self.combined_conv(i)
        c_forget, c_input, c_output, c_state = torch.split(combined, 4, dim=1)
        forget_gate = torch.sigmoid(c_forget)
        input_gate = torch.sigmoid(c_input)
        output_gate = torch.sigmoid(c_output)
        state_gate = torch.tanh(c_state)

        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)
        return cell, hidden
