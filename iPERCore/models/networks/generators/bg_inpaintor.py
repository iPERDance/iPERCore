# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim_out, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim_out, affine=False),
        )

    def forward(self, x):
        return x + self.main(x)


class ResNetInpaintor(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    def __init__(self, c_dim=4, num_filters=(64, 128, 256, 512), n_res_block=6):
        super(ResNetInpaintor, self).__init__()
        self._name = 'ResNetInpaintor'

        layers = list()
        layers.append(nn.Conv2d(c_dim, num_filters[0], kernel_size=7, stride=1, padding=3, bias=True))
        layers.append(nn.InstanceNorm2d(num_filters[0], affine=False))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        n_down = len(num_filters) - 1
        for i in range(1, n_down + 1):
            layers.append(nn.Conv2d(num_filters[i - 1], num_filters[i], kernel_size=3, stride=2, padding=1, bias=True))
            layers.append(nn.InstanceNorm2d(num_filters[i], affine=False))
            layers.append(nn.ReLU(inplace=True))

        # Bottleneck
        for i in range(n_res_block):
            layers.append(ResidualBlock(dim_in=num_filters[-1], dim_out=num_filters[-1]))

        # Up-Sampling
        for i in range(n_down, 0, -1):
            layers.append(nn.ConvTranspose2d(num_filters[i], num_filters[i - 1], kernel_size=4,
                                             stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(num_filters[i - 1], affine=False))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(num_filters[0], 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
