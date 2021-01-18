# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

from torch.utils.data import DataLoader

from .dataset import DatasetFactory


class CustomDatasetDataLoader(object):
    def __init__(self, opt, is_for_train=True):
        self._opt = opt
        self._is_for_train = is_for_train
        self._num_threds = opt.num_workers
        self._create_dataset()

    def _create_dataset(self):
        self._dataset = DatasetFactory.get_by_name(self._opt.dataset_mode, self._opt, self._is_for_train)
        self._dataloader = DataLoader(
            self._dataset,
            batch_size=self._opt.batch_size,
            shuffle=not self._opt.serial_batches,
            num_workers=int(self._num_threds),
            drop_last=True)

    def load_data(self):
        return self._dataloader

    def __len__(self):
        return len(self._dataset)

