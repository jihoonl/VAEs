import torch
from torch import nn
import torch.nn.functional as F


class TowerEncoder(nn.Module):

    def __init__(self, d, h, w, zdim=64, hdim=128, use_bn=True, *args,
                 **kwargs):
        super().__init__()
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(d, hdim, kernel_size=2, stride=2, bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(hdim, hdim, kernel_size=2, stride=2, bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim) if use_bn else nn.Identity())

        self.conv3 = nn.Sequential(
            nn.Conv2d(hdim,
                      hdim // 2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim // 2) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(hdim // 2, hdim, kernel_size=2, stride=2,
                      bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim) if use_bn else nn.Identity(),
        )
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = nn.Sequential(
            nn.Conv2d(hdim,
                      hdim,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim) if use_bn else nn.Identity())

        self.conv5 = nn.Sequential(
            nn.Conv2d(hdim,
                      hdim // 2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim // 2) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(hdim // 2,
                      hdim,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim) if use_bn else nn.Identity())
        self.conv6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hdim, hdim, kernel_size=1, stride=1, bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential(
            nn.Conv2d(hdim, zdim * 2, kernel_size=1, stride=1, bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim) if use_bn else nn.Identity(),
            nn.Sigmoid())

    def forward(self, x):
        out = self.stem(x)
        out = self.gaussian(out)
        return out

    def stem(self, x):
        skip_in = self.conv1(x)
        skip_out = self.conv2(skip_in)

        out = self.relu(skip_out + self.conv3(skip_in))
        out = self.conv6(self.conv4(skip_out) + self.conv5(out))
        return out

    def gaussian(self, h):
        return self.conv7(h)


class TowerDecoder(nn.Module):

    def __init__(self, d, h, w, zdim=64, hdim=128, use_bn=True, *args,
                 **kwargs):
        super().__init__()

        if 'odim' in kwargs:
            d = kwargs['odim']

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(zdim,
                               hdim // 2,
                               kernel_size=1,
                               stride=1,
                               bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim // 2) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(hdim // 2,
                               hdim,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim) if use_bn else nn.Identity())
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(hdim // 2,
                               hdim,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hdim,
                               hdim,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim) if use_bn else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(hdim,
                               hdim,
                               kernel_size=2,
                               stride=2,
                               bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim) if use_bn else nn.Identity())

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(hdim,
                               hdim // 2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim // 2) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hdim // 2,
                               hdim,
                               kernel_size=2,
                               stride=2,
                               bias=not use_bn),
            nn.BatchNorm2d(num_features=hdim) if use_bn else nn.Identity())
        self.deconv6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hdim,
                               d,
                               kernel_size=2,
                               stride=2,
                               bias=not use_bn))

    def forward(self, z):
        skip_in = self.deconv1(z)
        skip_out = self.deconv2(skip_in)

        out = self.relu(skip_out + self.deconv3(skip_in))
        out = self.deconv6(self.deconv4(out) + self.deconv5(out))

        return out
