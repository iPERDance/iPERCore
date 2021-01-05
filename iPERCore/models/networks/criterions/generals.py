# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, mat):
        return torch.mean(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.mean(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))


class TemporalSmoothLoss(nn.Module):
    def __init__(self):
        super(TemporalSmoothLoss, self).__init__()

    def forward(self, mat):
        return torch.mean(torch.abs(mat[:, 1:] - mat[:, 0:-1]))

