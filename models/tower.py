import torch
from torch import nn
import torch.nn.functional as F


class TowerEncoder(nn.Module):

    def __init__(self, d, h, w, zdim=64, hdim=128, *args, **kwargs):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(d, hdim, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(hdim, hdim, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(hdim,
                               hdim // 2,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv4 = nn.Conv2d(hdim // 2, hdim, kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(hdim, hdim, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(hdim,
                               hdim // 2,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv7 = nn.Conv2d(hdim // 2,
                               hdim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv8 = nn.Conv2d(hdim, zdim * 2, kernel_size=1, stride=1)

    def forward(self, x):
        skip_in = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        out = F.relu(self.conv3(skip_in))
        out = skip_out + F.relu(self.conv4(out))

        skip_out = F.relu(self.conv6(out))

        out = F.relu(self.conv7(skip_out)) + F.relu(self.conv5(out))
        out = F.relu(self.conv8(out))

        return out


class TowerDecoder(nn.Module):

    def __init__(self, d, h, w, zdim=64, hdim=128, *args, **kwargs):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(zdim,
                                          hdim // 2,
                                          kernel_size=1,
                                          stride=1)
        self.deconv2 = nn.ConvTranspose2d(hdim // 2,
                                          hdim,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        self.deconv3 = nn.ConvTranspose2d(hdim // 2,
                                          hdim,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        self.deconv4 = nn.ConvTranspose2d(hdim,
                                          hdim,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        self.deconv5 = nn.ConvTranspose2d(hdim, hdim, kernel_size=2, stride=2)
        self.deconv6 = nn.ConvTranspose2d(hdim,
                                          hdim // 2,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        self.deconv7 = nn.ConvTranspose2d(hdim // 2,
                                          hdim,
                                          kernel_size=2,
                                          stride=2)
        self.deconv8 = nn.ConvTranspose2d(hdim, d, kernel_size=2, stride=2)

    def forward(self, z):
        skip_in = F.relu(self.deconv1(z))
        skip_out = F.relu(self.deconv2(skip_in))

        out = F.relu(self.deconv3(skip_in))
        out = skip_out + F.relu(self.deconv4(out))

        skip_out = F.relu(self.deconv6(out))

        out = F.relu(self.deconv7(skip_out)) + F.relu(self.deconv5(out))
        out = torch.sigmoid(self.deconv8(out))

        return out
