# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os.path
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from .dataset import DatasetBase
from .transforms import ImageNormalizeToTensor


class PlaceDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(PlaceDataset, self).__init__(opt, is_for_train)
        self._name = 'PlaceDataset'
        self._read_dataset_paths()

    def _read_dataset_paths(self):
        if self._is_for_train:
            sub_folder = 'train'
        else:
            sub_folder = 'val'

        self._data_dir = os.path.join(self._opt.place_dir, sub_folder)

        self.dataset = datasets.ImageFolder(self._data_dir, transform=self._transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item][0]

    def _create_transform(self):
        image_size = self._opt.bg_size
        transforms.ToTensor()
        transform_list = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            ImageNormalizeToTensor()]
        self._transform = transforms.Compose(transform_list)



