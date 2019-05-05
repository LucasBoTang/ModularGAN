import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    """
    Residual Block with instance normalization
    """
    def __init__(self, dim=256):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.main(x))


class Encoder(nn.Module):
    """
    Encoder Module networks
    """
    def __init__(self, conv_dim=64, repeat_num=6):
        super(Encoder, self).__init__()

        layers = []

        # down-sampling layers
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        for i in range(2):
            layers.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(conv_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            conv_dim = conv_dim * 2

        # residual blocks layers
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim=conv_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Transformer(nn.Module):
    """
    Transformer Module networks
    """
    def __init__(self, conv_dim=256, c_dim=5, repeat_num=6):
        super(Transformer, self).__init__()

        layers = []

        # transform layer
        layers.append(nn.Conv2d(conv_dim+c_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # residual blocks layers
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim=conv_dim))

        self.main = nn.Sequential(*layers)

        # attention layer
        self.mask = nn.Sequential(
            nn.Conv2d(conv_dim, 1, kernel_size=7, stride=1, padding=3, bias=True),
            nn.Tanh())

    def forward(self, x, c):
        # replicate label spatially
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        # transformed feature map
        f = self.main(torch.cat((x, c), 1))
        # alpha mask
        g = (1 + self.mask(f)) / 2
        return  g * f + (1 - g) * x


class Reconstructor(nn.Module):
    """
    Reconstructor Module networks
    """
    def __init__(self, conv_dim=256):
        super(Reconstructor, self).__init__()

        layers = []

        # up-sampling layers
        layers.append(nn.ConvTranspose2d(conv_dim, conv_dim//2, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim//2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        conv_dim = conv_dim // 2

        layers.append(nn.ConvTranspose2d(conv_dim, conv_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim//2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        conv_dim = conv_dim // 2

        # convlutional layer
        layers.append(nn.Conv2d(conv_dim, 3, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(3, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """
    Discriminator network with PatchGAN
    """
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []

        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            conv_dim = conv_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(conv_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(conv_dim, c_dim, kernel_size=image_size//2**repeat_num, bias=False)

    def forward(self, x):
        h = self.main(x)
        # real or fake on each patch
        out_src = self.conv1(h)
        # predicts attributes
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
