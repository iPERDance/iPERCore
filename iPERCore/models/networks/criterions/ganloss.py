# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
import torch.nn as nn


class LSGANLoss(nn.Module):
    def __init__(self):
        super(LSGANLoss, self).__init__()

    def forward(self, x, y):
        if not isinstance(x, list):
            x = [x]

        loss = 0.0
        num = len(x)
        for out in x:
            loss += torch.mean((out - y) ** 2)

        loss /= num
        return loss
