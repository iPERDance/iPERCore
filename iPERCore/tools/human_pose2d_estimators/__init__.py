# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import abc
import numpy as np


class BasePoseEstimator(object):

    def boxes_from_keypoints(self, kps, threshold=0.3):
        """

        Args:
            kps (np.ndarray): (number of keypoints, 3)
            threshold (float):

        Returns:

        """

        valid = kps[:, 2] > threshold
        valid_kps = kps[valid, 0:2]

        min_xy = np.min(valid_kps, axis=0, keepdims=False)
        max_xy = np.max(valid_kps, axis=0, keepdims=False)
        boxes = np.concatenate([min_xy, max_xy], axis=0)

        return boxes

    @abc.abstractmethod
    def run_single_image(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def run_over_folder(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def format_for_smplify(self, pose_results):
        pass


def build_pose2d_estimator(name: str, *args, **kwargs):
    if name == "caffe_openpose":
        from .openpose.caffe_runner import CaffeOpenPoseEstimator
        model = CaffeOpenPoseEstimator(*args, **kwargs)

    elif name == "openpose":
        from .openpose.runner import OpenPoseRunner
        model = OpenPoseRunner(*args, **kwargs)

    else:
        raise ValueError(name)

    return model
