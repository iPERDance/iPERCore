# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .transforms import ToTensor, ImageTransformer


class DatasetFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, is_for_train):
        """

        Args:
            dataset_name:
            opt:
            is_for_train:

        Returns:
            dataset (torch.utils.data.Dataset): it must implements the function of self.__getitem(item), and it
                will return a sample dictionary which contains the following information:
                --images (torch.Tensor): (ns + nt, 3, h, w), here `ns` and `nt` are the number of source and targets;
                --masks (torch.Tensor): (ns + nt, 1, h, w);
                --smpls (torch.Tensor): (ns + nt, 85);
                --bg (torch.Tensor): (3, h, w).
        """

        if dataset_name == "ProcessedVideo":
            from .processed_video_dataset import ProcessedVideoDataset
            dataset = ProcessedVideoDataset(opt, is_for_train)

        elif dataset_name == "ProcessedVideo+Place2":
            from .concat_dataset import ProcessedVideoPlace2Dataset
            dataset = ProcessedVideoPlace2Dataset(opt, is_for_train)

        else:
            raise ValueError(f"Dataset {dataset_name} not recognized.")

        print(f"Dataset {dataset.name} was created.")
        return dataset


class DatasetBase(Dataset):
    def __init__(self, opt, is_for_train):
        super(DatasetBase, self).__init__()
        self._name = "BaseDataset"
        self._opt = opt
        self._is_for_train = is_for_train
        self._intervals = opt.intervals
        self._create_transform()

    @property
    def name(self):
        return self._name

    def _create_transform(self):
        self._transform = transforms.Compose([])

    def get_transform(self):
        return self._transform

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class VideoDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(VideoDataset, self).__init__(opt, is_for_train)
        self._name = "VideoDataset"

        self._num_videos = 0
        self._vids_info = []
        self._dataset_size = 0
        self._intervals = opt.intervals

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        """

        Args:
            index (int): the sample index of self._dataset_size.

        Returns:
            sample (dict): the data sample, it contains the following informations:
                --images (torch.Tensor): (ns + nt, 3, h, w), here `ns` and `nt` are the number of source and targets;
                --masks (torch.Tensor): (ns + nt, 1, h, w);
                --smpls (torch.Tensor): (ns + nt, 85);

        """
        pass

    def _read_vids_info(self):
        pass

    def _load_pairs(self, vid_info):
        pass

    def _create_transform(self):
        transform_list = [
            ImageTransformer(output_size=self._opt.image_size),
            ToTensor()]
        self._transform = transforms.Compose(transform_list)

    @property
    def video_info(self):
        return self._vids_info
