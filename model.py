import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block with instance normalization
    """

    def __init__(self, dim=256):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return x + self.main(x)


class Encoder(nn.Module):
    """
    Encoder Module networks
    """

    def __init__(self, conv_dim=64, repeat_num=6):
        super(Encoder, self).__init__()

        layers = []

        # down-sampling layers
        layers.append(
            nn.Conv2d(3,
                      conv_dim,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False))
        layers.append(
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        for i in range(2):
            layers.append(
                nn.Conv2d(
                    conv_dim,
                    conv_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ))
            layers.append(
                nn.InstanceNorm2d(conv_dim * 2,
                                  affine=True,
                                  track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            conv_dim *= 2

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
        layers.append(
            nn.Conv2d(
                conv_dim + c_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ))
        layers.append(
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # residual blocks layers
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim=conv_dim))

        self.main = nn.Sequential(*layers)

        # attention layer
        self.attention = nn.Sequential(
            nn.Conv2d(conv_dim,
                      1,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False),
            nn.Tanh(),
        )

    def forward(self, x, c):
        # replicate label spatially
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        # transformed feature map
        f = self.main(torch.cat((x, c), dim=1))
        # alpha mask
        g = (1 + self.attention(f)) / 2
        return g * f + (1 - g) * x


class Reconstructor(nn.Module):
    """
    Reconstructor Module networks
    """

    def __init__(self, conv_dim=256):
        super(Reconstructor, self).__init__()

        layers = []

        # up-sampling layers
        for _ in range(2):
            layers.append(
                nn.ConvTranspose2d(
                    conv_dim,
                    conv_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ))
            layers.append(
                nn.InstanceNorm2d(conv_dim // 2,
                                  affine=True,
                                  track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            conv_dim //= 2

        # convlutional layer
        layers.append(
            nn.Conv2d(conv_dim,
                      3,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False))
        layers.append(nn.Tanh())

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
        layers.append(
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        for i in range(1, repeat_num):
            layers.append(
                nn.Conv2d(conv_dim,
                          conv_dim * 2,
                          kernel_size=4,
                          stride=2,
                          padding=1))
            layers.append(nn.LeakyReLU(0.01))
            conv_dim *= 2

        self.main = nn.Sequential(*layers)
        self.out_src = nn.Conv2d(conv_dim,
                                 1,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=False)
        self.out_cls = nn.Conv2d(conv_dim,
                                 c_dim,
                                 kernel_size=image_size // 2**repeat_num,
                                 bias=False)

    def forward(self, x):
        h = self.main(x)
        # patch gan classification
        out_src = self.out_src(h)
        # attributes classification
        out_cls = self.out_cls(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


if __name__ == "__main__":
    """
    test code
    """
    # model structure
    print("Encoder")
    E = Encoder().cuda()
    print(E)
    print("\n")

    print("Transformer")
    T = Transformer().cuda()
    print(T)
    print("\n")

    print("Reconstructor")
    R = Reconstructor().cuda()
    print(R)
    print("\n")

    print("Discriminator")
    D = Discriminator().cuda()
    print(D)
    print("\n")

    # tensor flow
    x = torch.randn(8, 3, 128, 128).cuda()
    c = torch.randn(8, 5).cuda()
    print("The size of input image: {}".format(x.size()))
    print("The size of input label: {}".format(c.size()))

    out = E(x)
    print("The size of Encoder output: {}".format(out.size()))

    out = T(out, c)
    print("The size of Transformer output: {}".format(out.size()))

    out = R(out)
    print("The size of Reconstructor output: {}".format(out.size()))

    out_src, out_cls = D(x)
    print("The size of src out: {}".format(out_src.size()))
    print("The size of cls out: {}".format(out_cls.size()))
