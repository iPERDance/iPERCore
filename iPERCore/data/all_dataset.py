# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

from torch.utils.data import ConcatDataset
import numpy as np
import os.path

from .dataset import DatasetBase
from .iPER_dataset import IPERDataset, SeqIPERDataset
from .fashion_dataset import FashionVideoDataset, SeqFashionVideoDataset
from .place_dataset import PlaceDataset
from .motion_synthetic_dataset import MotionSyntheticDataset, SeqMotionSyntheticDataset


class ConcatVideoDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(ConcatVideoDataset, self).__init__(opt, is_for_train)
        self._name = "ConcatVideoDataset"

        dataset_mode = opt.dataset_mode
        all_datasets = []

        if "Seq" in dataset_mode:
            if os.path.exists(opt.iPER_dir):
                all_datasets.append(SeqIPERDataset(opt, is_for_train))

            if opt.motion_synthetic_dir and os.path.exists(opt.motion_synthetic_dir):
                all_datasets.append(SeqMotionSyntheticDataset(opt, is_for_train))

        else:
            if os.path.exists(opt.iPER_dir):
                all_datasets.append(IPERDataset(opt, is_for_train))

            if opt.motion_synthetic_dir and os.path.exists(opt.motion_synthetic_dir):
                all_datasets.append(MotionSyntheticDataset(opt, is_for_train))

        self.all_datasets = ConcatDataset(all_datasets)

    def __len__(self):
        return len(self.all_datasets)

    def __getitem__(self, item):
        return self.all_datasets[item]


class ConcatVideoPlaceDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(ConcatVideoPlaceDataset, self).__init__(opt, is_for_train)
        self._name = "ConcatVideoPlaceDataset"

        dataset_mode = opt.dataset_mode
        all_datasets = []

        if "Seq" in dataset_mode:
            if opt.iPER_dir and os.path.exists(opt.iPER_dir):
                all_datasets.append(SeqIPERDataset(opt, is_for_train))

            if opt.fashion_dir and os.path.exists(opt.fashion_dir):
                all_datasets.append(SeqFashionVideoDataset(opt, is_for_train))

            if opt.motion_synthetic_dir and os.path.exists(opt.motion_synthetic_dir):
                all_datasets.append(SeqMotionSyntheticDataset(opt, is_for_train))

        else:
            if opt.iPER_dir and os.path.exists(opt.iPER_dir):
                all_datasets.append(IPERDataset(opt, is_for_train))

            if opt.fashion_dir and os.path.exists(opt.fashion_dir):
                all_datasets.append(FashionVideoDataset(opt, is_for_train))

            if opt.motion_synthetic_dir and os.path.exists(opt.motion_synthetic_dir):
                all_datasets.append(MotionSyntheticDataset(opt, is_for_train))

        self.all_datasets = ConcatDataset(all_datasets)
        self.place = PlaceDataset(opt, is_for_train)
        self._dataset_size = len(self.all_datasets)
        self._num_places = len(self.place)
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
        sample = self.all_datasets[item]

        if self._place_need_mode:
            place_ids = item % self._num_places
        else:
            place_ids = self.sample_ids[item]

        aug_bg = self.place[place_ids]

        sample["bg"] = aug_bg
        return sample




