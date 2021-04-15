# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import numpy as np

from .dataset import DatasetBase
from .processed_video_dataset import ProcessedVideoDataset
from .place_dataset import Place2Dataset


class ProcessedVideoPlace2Dataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(ProcessedVideoPlace2Dataset, self).__init__(opt, is_for_train)
        self._name = "ProcessedVideoPlace2Dataset"

        self.video_dataset = ProcessedVideoDataset(opt, is_for_train)
        self.place_dataset = Place2Dataset(opt, is_for_train)
        self._dataset_size = len(self.video_dataset)
        self._num_places = len(self.place_dataset)
        if self._num_places <= self._dataset_size:
            self._place_need_mode = True
            self.sample_ids = np.arange(0, self._num_places)
        else:
            self._place_need_mode = False
            interval = self._num_places // self._dataset_size
            self.sample_ids = np.arange(0, self._num_places, interval)[0:self._dataset_size]

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, item):
        sample = self.video_dataset[item]

        if self._place_need_mode:
            place_ids = item % self._num_places
        else:
            place_ids = self.sample_ids[item]

        aug_bg = self.place_dataset[place_ids]

        sample["bg"] = aug_bg
        return sample




