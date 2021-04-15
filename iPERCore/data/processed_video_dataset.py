# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import numpy as np
from tqdm import tqdm

from iPERCore.tools.utils.filesio import cv_utils
from iPERCore.tools.utils.filesio.persistence import load_pickle_file
from iPERCore.services.options.process_info import read_src_infos

from .dataset import VideoDataset


class ProcessedVideoDataset(VideoDataset):

    def __init__(self, opt, is_for_train):
        super(ProcessedVideoDataset, self).__init__(opt, is_for_train)

        self._read_vids_info()

    def _read_single_dataset(self, data_dir, txt_path):

        with open(txt_path, "r") as reader:

            for line in tqdm(reader.readlines()):
                vid_name = line.rstrip()
                vid_info_path = os.path.join(data_dir, "primitives", vid_name, "processed", "vid_info.pkl")

                # print(vid_info_path)
                vid_info = load_pickle_file(vid_info_path)
                vid_info = read_src_infos(vid_info, num_source=self._opt.num_source, ignore_bg=True)

                self._vids_info.append(vid_info)
                self._num_videos += 1
                self._dataset_size += vid_info["length"]

    def _read_vids_info(self):

        data_dir_list = self._opt.dataset_dirs
        for data_dir in data_dir_list:

            if self._is_for_train:
                txt_path = os.path.join(data_dir, "train.txt")
            else:
                txt_path = os.path.join(data_dir, "val.txt")

            self._read_single_dataset(data_dir, txt_path)

    def _load_pairs(self, vid_info):
        ns = self._opt.num_source

        length = vid_info["length"]
        ft_ids = vid_info["ft_ids"]

        replace = ns >= len(ft_ids)
        src_ids = list(np.random.choice(ft_ids, ns, replace=replace))
        src_ids[0] = ft_ids[0]

        tsf_ids = list(np.random.choice(length, self._opt.time_step, replace=False))
        tsf_ids.sort()

        # take the source and target ids
        pair_ids = src_ids + tsf_ids
        smpls = vid_info["smpls"][pair_ids]

        images = []
        masks = []
        image_dir = vid_info["img_dir"]
        images_names = vid_info["images"]
        alphas_paths = vid_info["alpha_paths"]

        for t in pair_ids:
            image_path = os.path.join(image_dir, images_names[t])
            image = cv_utils.read_cv2_img(image_path)

            images.append(image)

            mask = cv_utils.read_mask(alphas_paths[t], self._opt.image_size)

            # front is 0, and background is 1
            mask = 1.0 - mask
            masks.append(mask)

        return images, smpls, masks

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

        vid_info = self._vids_info[index % self._num_videos]

        images, smpls, masks = self._load_pairs(vid_info)

        # pack data
        sample = {
            "images": images,
            "smpls": smpls,
            "masks": masks
        }

        sample = self._transform(sample)

        return sample
