# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
import os.path
import numpy as np

from iPERCore.tools.utils.filesio import cv_utils
from iPERCore.services.options.process_info import ProcessInfo

from .dataset import VideoDataset


class PersonalizedDataset(VideoDataset):

    def __init__(self, opt, meta_process_list):
        """

        Args:
            opt:
            meta_process_list (list of MetaProcess):
        """

        super(VideoDataset, self).__init__(opt, is_for_train=False)
        self._name = "PersonalizedDataset"

        # read dataset
        self._share_bg = opt.share_bg
        self._dataset_size = 0
        self._num_videos = 0
        self._meta_process_list = meta_process_list
        self._read_vids_info()

    @property
    def num_videos(self):
        return self._num_videos

    def size(self):
        return self._dataset_size

    def __len__(self):
        # TODO, when the dataloader runs over a dataset, it will create process for dataloader. When the
        #  dataset is small, this will increase the overhead at each step for the management of process.
        #  So, here, when the dataset is small, we set it to 1000.
        #  Thus, we need to carefully control the termination in the training loop .
        return max(1000, self._dataset_size)

    def _read_vids_info(self):
        vid_infos_list = []

        for meta_process in self._meta_process_list:
            # print(vid_dir)
            process_info = ProcessInfo(meta_process)
            process_info.deserialize()
            vid_info = process_info.convert_to_src_info(self._opt.num_source)

            length = vid_info["length"]

            if length == 0:
                continue

            vid_info["probs"] = self._sample_probs(vid_info)

            vid_infos_list.append(vid_info)
            self._num_videos += 1
            self._dataset_size += vid_info["length"]

        self._vids_info = vid_infos_list

    def _sample_probs(self, vid_info, ft=4, bk=2):
        length = vid_info["length"]
        ft_ids = vid_info["ft_ids"][0]
        bk_ids = vid_info["bk_ids"][0]

        if length == 1:
            probs = np.array([1])
        else:
            pi = 1 / (ft + bk + length - 2)
            probs = np.zeros((length,)) + pi

            probs[ft_ids] = pi * ft
            probs[bk_ids] = pi * bk

        return probs

    def _load_pairs(self, index):
        vid_info = self._vids_info[index % self._num_videos]
        length = vid_info["length"]
        img_dir = vid_info["img_dir"]
        src_ids = vid_info["src_ids"]
        offsets = vid_info["offsets"]
        alpha_paths = vid_info["alpha_paths"]
        replaced_paths = vid_info["replaced_paths"]
        actual_bg_path = vid_info["actual_bg_path"]
        bg_dir = vid_info["bg_dir"]

        if length < self._opt.time_step:
            replace = True
        else:
            replace = False

        tsf_ids = list(np.random.choice(length, self._opt.time_step, replace=replace))
        pair_ids = src_ids + tsf_ids

        # print(pair_ids, vid_info["probs"])

        # print(len(pair_ids), src_ids, pair_ids)
        # print(pair_ids, np.random.random())

        # if length > 1:
        #     tsf_ids = [src_ids[0]] + [index]
        #     pair_ids = src_ids + tsf_ids
        # else:
        #     pair_ids = [0, 0]

        smpls = vid_info["smpls"][pair_ids]

        images_name = vid_info["images"]
        images = []
        masks = []
        pseudo_bgs = []

        for t in pair_ids:
            name = images_name[t]
            image_path = os.path.join(img_dir, name)
            image = cv_utils.read_cv2_img(image_path)
            images.append(image)

            mask = cv_utils.read_mask(alpha_paths[t], self._opt.image_size)

            # attention! Here, the front is 0, and the background is 1
            mask = 1.0 - mask
            masks.append(mask)

        if self._share_bg:
            if actual_bg_path is not None:
                bg_path = actual_bg_path
            else:
                bg_path = np.random.choice(replaced_paths)
            bg_img = cv_utils.read_cv2_img(bg_path)
            bg_img = cv_utils.normalize_img(bg_img, image_size=self._opt.image_size, transpose=True)
            pseudo_bgs = bg_img
        else:
            if actual_bg_path is not None:
                bg_img_paths = [actual_bg_path] * len(src_ids)
            else:
                bg_img_paths = []
                for s_id in src_ids:
                    name = images_name[s_id]
                    bg_name = name.split(".")[0] + "_replaced.png"
                    bg_path = os.path.join(bg_dir, bg_name)
                    bg_img_paths.append(bg_path)

            for bg_path in bg_img_paths:
                bg_img = cv_utils.read_cv2_img(bg_path)
                bg_img = cv_utils.normalize_img(bg_img, image_size=self._opt.image_size, transpose=True)
                pseudo_bgs.append(bg_img)
            pseudo_bgs = np.stack(pseudo_bgs)

        links_ids = vid_info["links"]

        return images, smpls, masks, offsets, pseudo_bgs, links_ids

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:

        """

        # assert (index < self._dataset_size)

        # start_time = time.time()
        # get sample data
        images, smpls, masks, offsets, pseudo_bgs, links_ids = self._load_pairs(index)

        # pack data
        sample = {
            "images": images,
            "smpls": smpls,
        }

        sample = self._transform(sample)
        sample["masks"] = torch.tensor(masks).float()
        sample["bg"] = torch.tensor(pseudo_bgs).float()
        sample["offsets"] = torch.from_numpy(offsets.astype(np.float32, copy=False))
        sample["links_ids"] = torch.tensor(links_ids).long()

        return sample
