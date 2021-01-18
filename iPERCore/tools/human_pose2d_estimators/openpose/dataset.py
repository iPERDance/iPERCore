# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import numpy as np
import math
import cv2
import os


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def preprocess(img, net_input_height_size=368, stride=8,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    """

    Args:
        img:

    Returns:

    """

    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    outputs = {
        "img": padded_img,
        "pad": pad,
        "scale": scale
    }

    return outputs


class ImageFolderDataset(object):
    def __init__(self, root_dir, valid_names=None):

        if valid_names is None:
            img_names = os.listdir(root_dir)
            img_names.sort()
        else:
            img_names = valid_names

        self.root_dir = root_dir
        self.img_names = img_names
        self.file_paths = [os.path.join(root_dir, img_name) for img_name in img_names]
        self.max_idx = len(img_names)
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __len__(self):
        return self.max_idx

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration

        img = cv2.imread(self.file_paths[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_paths[self.idx]))
        self.idx = self.idx + 1
        return img


