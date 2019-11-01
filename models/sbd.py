import torch
from torch import nn
from torch.nn import functional as F
from .tower import TowerEncoder


class SpatialBroadcastDecoder(nn.Module):
    """
    Spatial Broadcast Decoder: A Simple Architecture for Disentangled Representations in VAEs
    -> https://arxiv.org/pdf/1901.07017.pdf

    And Decoder conv layer intoruced in IODINE
    -> Multi-Object Representation Learning with Iterative Variational Inference
    -> https://arxiv.org/pdf/1903.00450.pdf
    """

    def __init__(self, d, h, w, zdim=7, hdim=64, *args, **kwargs):
        super().__init__()

        self.image_shape = (h, w)

        x_range = torch.linspace(-1, 1, w)
        y_range = torch.linspace(-1, 1, h)

        x_grid, y_grid = torch.meshgrid([x_range, y_range])
        self.register_buffer('x_grid', x_grid.view(1, 1, *x_grid.shape))
        self.register_buffer('y_grid', y_grid.view(1, 1, *y_grid.shape))

        # Decoder model in IODINE page 13.
        self.conv1 = nn.Conv2d(zdim + 2,
                               hdim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(hdim, hdim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hdim, hdim, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(hdim, hdim, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(hdim, d, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        batch, *zdims = z.shape

        # Spatial Broadcasting
        z_broad = z.view(batch, -1)
        z_broad = z.view(*z_broad.shape, 1, 1)
        z_broad = z_broad.expand(-1, -1, *self.image_shape)
        broadcasted = torch.cat((z_broad, self.x_grid.expand(
            (batch, -1, -1, -1)), self.y_grid.expand((batch, -1, -1, -1))),
                                dim=1)

        # Decoding
        out = F.elu(self.conv1(broadcasted))
        out = F.elu(self.conv2(out))
        out = F.elu(self.conv2(out))
        out = F.elu(self.conv3(out))
        out = F.elu(self.conv4(out))
        out = torch.sigmoid(self.conv5(out))
        return out
