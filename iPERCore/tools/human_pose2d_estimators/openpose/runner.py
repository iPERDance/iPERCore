# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import numpy as np
import torch
from easydict import EasyDict
from tqdm import tqdm

from iPERCore.tools.utils.filesio.persistence import load_toml_file

from .models import build_openpose_model
from .post_process import infer, infer_post_process, infer_fast, infer_fast_post_process
from .dataset import ImageFolderDataset
from ..utils.pose_utils import POSE_CLASS
from .. import BasePoseEstimator


class OpenPoseRunner(BasePoseEstimator):

    def __init__(self,
                 cfg_or_path,
                 tracker=None,
                 device=torch.device("cpu")):
        """

        Args:
            cfg_or_path: the configuration 
            tracker:
            device:
        """

        if isinstance(cfg_or_path, str):
            cfg = load_toml_file(cfg_or_path)
            cfg = EasyDict(cfg)
        else:
            cfg = cfg_or_path

        self.cfg = cfg
        self.multi_scales = cfg["data"]["multi_scales"]
        self.stride = cfg["data"]["stride"]
        self.upsample_ratio = cfg["data"]["upsample_ratio"]

        self.PoseClass = POSE_CLASS[cfg["model"]["name"]]

        self.tracker = tracker
        self.device = device

        model = build_openpose_model(cfg["model"]["name"])
        checkpoints = torch.load(cfg["model"]["ckpt_path"], map_location="cpu")
        model.load_state_dict(checkpoints)

        self.model = model.to(self.device)
        self.model.eval()

    def format_output(self, current_poses, person_ids):
        """

        Args:
            current_poses:
            person_ids:

        Returns:

        """
        number = len(current_poses)
        has_person = number > 0

        output = {
            "number": number,
            "has_person": has_person,
        }

        if number == 0:
            output["boxes_XYXY"] = None

            output["keypoints"] = {
                "pose_keypoints_2d": [],
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": []
            }

        else:
            kps = current_poses[person_ids].keypoints
            output["boxes_XYXY"] = current_poses[person_ids].bbox

            # print(output["boxes_XYXY"], keypoints_info[0]["bbox"])

            output["keypoints"] = {
                "pose_keypoints_2d": kps,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": []
            }

        return output

    def run_single_image(self, img, height_size=368, use_fast=True, previous_poses=None, pose_smooth=False):
        """

        Args:
            img (np.darary):
            height_size (int):
            use_fast (bool):
            previous_poses (None or BasePose):
            pose_smooth (bool):
        Returns:
            output (dict): the dict information of, and it contains:
                --number (int): the number of the person;
                --has_person (bool): is there person or not, e.g number > 0;
                --boxes_XYXY (np.ndarray): (4,), the bounding boxes in the range of [0, height/width];
                --orig_shape (tuple): (height, width), the original shape of the input image;

                --keypoints (dict):
                    --pose_keypoints_2d (np.ndarray): in the range of [0, height/width]
                    --face_keypoints_2d (np.ndarray or List): in the range of [0, height/width]
                    --hand_left_keypoints_2d (np.ndarray or List): in the range of [0, height/width]
                    --hand_right_keypoints_2d (np.ndarray or List): in the range of [0, height/width]
        """

        if use_fast:
            net_outs = infer_fast(
                self.model, img, height_size, self.stride, self.upsample_ratio, self.device
            )
            # ipdb.set_trace()
            output = infer_fast_post_process(net_outs, self.PoseClass)

        else:
            net_outs = infer(
                self.model, img, self.multi_scales, height_size, self.stride, self.device
            )
            output = infer_post_process(net_outs, self.PoseClass)

        current_poses = output["current_poses"]

        if previous_poses is not None:
            current_poses = self.PoseClass.track_poses(previous_poses, current_poses, smooth=pose_smooth)

        # run tracker to get the target person ids
        if len(current_poses) > 1:
            person_bboxes = np.array([pose.bbox for pose in current_poses])
            _, person_ids = self.tracker(img, person_bboxes)
        else:
            person_ids = 0

        output = self.format_output(current_poses, person_ids)
        output["orig_shape"] = img.shape[0:2]

        if pose_smooth:
            return output, current_poses
        else:
            return output

    def run_over_folder(self, root_dir, height_size=368, use_fast=True, pose_smooth=True):
        """

        Run detector over all the images in one folder.

        Args:
            root_dir (str): the directory of the folder;
            height_size (int):
            use_fast (bool):
            pose_smooth (bool):

        Returns:
            pose_results (dict): the pose results and it contains the followings,
                --keypoints (list of dict): each dict information of, and it contains:
                    --number (int): the number of the person;
                    --has_person (bool): is there person or not, e.g number > 0;
                    --boxes_XYXY (np.ndarray): (4,), the bounding boxes in the range of [0, height/width];
                    --orig_shape (tuple): (height, width), the original shape of the input image;

                    --keypoints (dict):
                        --pose_keypoints_2d (np.ndarray or list): (23, 3) is in the range of [0, height/width];
                        --face_keypoints_2d (np.ndarray or list): (68, 3) is in the range of [0, height/width];
                        --hand_left_keypoints_2d (np.ndarray or list): (21, 3) is in the range of [0, height/width];
                        --hand_right_keypoints_2d (np.ndarray or list): (21, 3) is in the range of [0, height/width].

                --img_names (list of str): all image names whose boxes is not None.
                    Also, len(img_names) <= all images on the folder.
        """

        dataloader = ImageFolderDataset(root_dir=root_dir, valid_names=None)
        all_img_names = dataloader.img_names

        all_keypoints = []

        previous_poses = None
        for img in tqdm(dataloader):
            output, current_poses = self.run_single_image(
                img, height_size, use_fast, previous_poses, pose_smooth=True
            )

            if pose_smooth:
                previous_poses = current_poses

            all_keypoints.append(output)

        pose_results = {
            "keypoints": all_keypoints,
            "img_names": all_img_names
        }

        return pose_results

    def format_for_smplify(self, pose_results):
        pass
