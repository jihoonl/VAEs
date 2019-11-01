class TowerEncoder(nn.Module):

    def __init__(self, d, h, w, zdim=64, hdim=128, *args, **kwargs):
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
