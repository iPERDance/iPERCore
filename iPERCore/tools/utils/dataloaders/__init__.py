# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

from torch.utils.data import DataLoader


def build_inference_loader(dataset, batch_size=16, num_workers=4):
    inference_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )

    return inference_loader
