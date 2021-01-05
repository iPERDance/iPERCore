# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import cv2
import numpy as np
from typing import Union


def calculate_area(kps):
    """

    Args:
        kps (np.ndarray): (number of keypoints, 3).

    Returns:
        area (float): the area of the bounding boxes;
        boxes (list): (x0, y0, x1, y1)
    """

    vis = kps[:, 2] > 0

    if len(vis) > 0:
        vis_kps = kps[vis, 0:2]

        x0, y0 = np.min(vis_kps, axis=0)
        x1, y1 = np.max(vis_kps, axis=0)
        area = (x1 - x0) * (y1 - y0)
        boxes = [x0, y0, x1, y1]
    else:
        area = 0
        boxes = None

    return area, boxes


class CaffeOpenPoseEstimator(object):
    def __init__(self, config_path="/root/openpose/models", model_pose="BODY_25",
                 tracker=None, use_hand=False, use_face=False, device="cpu"):
        from openpose import pyopenpose as op

        params = dict()

        params["model_folder"] = config_path
        params["model_pose"] = model_pose
        params["keypoint_scale"] = 0  # 0 source resolution, 3 is in the range of [0, 1], 4 is in the range of [-1, 1]

        if use_hand:
            params["hand"] = use_hand
        if use_face:
            params["face"] = use_face

        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()

        self.opWarrper = opWrapper
        self.datum = datum
        self.use_hand = use_hand
        self.use_face = use_face

        self.tracker = tracker
        self.device = device
        self.has_detector = False

    def run_single_image(self, image_or_path: Union[np.ndarray, str], get_visual: bool = False):
        """

        Run the OpenPose to estimate the keypoints, here we return the largest area of person.

        Args:
            image_or_path (str or np.ndarray): if it is np.ndarray, it must be (height, width, 3) with np.uint8 in the
                range of [0, 255], BGR channel;
            get_visual (bool): return the visual output or not.

        Returns:
            output (dict): the information of output, and it contains:
                --number (int):
                --has_person (bool):
                --boxes_XYXY (tuple):
                --orig_shape (tuple): (height, width);
                --keypoints (dict):
                    --pose_keypoints_2d (np.ndarray):
                    --face_keypoints_2d (np.ndarray):
                    --hand_left_keypoints_2d (np.ndarray):
                    --hand_right_keypoints_2d (np.ndarray):
                    --visual (np.ndarray or None): (height, width, 3)
        """
        if isinstance(image_or_path, str):
            imageToProcess = cv2.imread(image_or_path)
        else:
            imageToProcess = np.copy(image_or_path)

        self.datum.cvInputData = imageToProcess
        self.opWarrper.emplaceAndPop([self.datum])

        output = self.post_process(self.datum, imageToProcess.shape[0:2])

        if get_visual:
            output["visual"] = self.datum.cvOutputData

        return output

    def post_process(self, datum, img_shape):
        """

        Args:
            datum:
            img_shape (tuple): (height, width)

        Returns:

        """
        height, width = img_shape

        poseKps = datum.poseKeypoints
        faceKps = datum.faceKeypoints
        leftHandKps = datum.handKeypoints[0]
        rightHandKps = datum.handKeypoints[1]

        if len(poseKps.shape) == 0:
            number = 0
            boxes = None
            poseKps = []
        else:
            number = len(poseKps)
            poseKps, leftHandKps, rightHandKps, faceKps, boxes = self.choose_large_one(
                poseKps, leftHandKps, rightHandKps, faceKps
            )

        output = self.format_output(number, boxes, poseKps, leftHandKps, rightHandKps, faceKps, height, width)
        return output

    def format_output(self, number, boxes, poseKps, leftHandKps, rightHandKps, faceKps, height, width):
        """

        Args:
            number (int):
            poseKps (np.ndarray):
            leftHandKps (np.ndarray ir List):
            rightHandKps (np.ndarray or List):
            faceKps (np.ndarray or List):
            height (int or None):
            width (int or None):

        Returns:
            output (dict): the dict information of, and it contains:
                --number (int):
                --has_person (bool):
                --boxes (tuple): the bounding boxes in the range of [0, height/width]
                --orig_shape (tuple): (height, width)

                --keypoints (dict):
                    --pose_keypoints_2d (np.ndarray): in the range of [0, height/width]
                    --face_keypoints_2d (np.ndarray or List): in the range of [0, height/width]
                    --hand_left_keypoints_2d (np.ndarray or List): in the range of [0, height/width]
                    --hand_right_keypoints_2d (np.ndarray or List): in the range of [0, height/width]

        """

        output = {
            "number": number,
            "has_person": number > 0,
            "boxes_XYXY": boxes,
            "orig_shape": (height, width),
            "keypoints": {
                "pose_keypoints_2d": poseKps,
                "face_keypoints_2d": faceKps,
                "hand_left_keypoints_2d": leftHandKps,
                "hand_right_keypoints_2d": rightHandKps,
            }
        }

        return output

    def choose_large_one(self, poseKps, leftHandKps, rightHandKps, faceKps):
        """

        Args:
            poseKps (np.ndarray): (bs, dim), if self.model_pose == "BODY_25", then dim = 75.
            leftHandKps:
            rightHandKps:
            faceKps:

        Returns:

        """
        bs = poseKps.shape[0]
        poseKps = np.reshape(poseKps, (bs, -1, 3))

        if self.use_hand:
            leftHandKps = np.reshape(leftHandKps, (bs, -1, 3))
            rightHandKps = np.reshape(rightHandKps, (bs, -1, 3))
            kps = np.concatenate([poseKps, leftHandKps, rightHandKps], axis=1)
        else:
            kps = poseKps

        max_boxes, max_area, ids = None, 0, 0

        for i in range(bs):
            area, boxes = calculate_area(kps[i])
            if area > max_area:
                max_area = area
                ids = i
                max_boxes = boxes

        max_poseKps = poseKps[ids]

        max_leftHandKps = []
        max_rightHandKps = []
        max_faceKps = []

        if self.use_hand:
            if len(leftHandKps) >= ids + 1:
                max_leftHandKps = leftHandKps[ids]

            if len(rightHandKps) >= ids + 1:
                max_rightHandKps = rightHandKps[ids]

        if self.use_face and len(faceKps) >= ids + 1:
            max_faceKps = faceKps[ids]

        return max_poseKps, max_leftHandKps, max_rightHandKps, max_faceKps, max_boxes

    def start(self):
        from openpose import pyopenpose as op
        if self.datum is None:
            self.datum = op.Datum()

    def close(self):
        self.datum = None
        self.opWarrper.stop()


if __name__ == "__main__":
    image_paths = [
        "/p300/tpami/datasets/Youtube-Dancer-18/processed/2rJvM52eOOc_1.mp4/"

    ]

    import glob
    import os
    import time
    from tqdm import tqdm
    from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer

    # define visualizer
    visualizer = VisdomVisualizer(
        env='debug',
        ip='http://10.10.10.100', port=31102
    )

    image_paths = glob.glob(
        os.path.join("/p300/tpami/datasets/Youtube-Dancer-18/processed/2rJvM52eOOc_1.mp4/images", "*"))

    image_paths.sort()

    openpose_estimator = CaffeOpenPoseEstimator(use_face=False, use_hand=True)

    for img_path in tqdm(image_paths):
        output = openpose_estimator.run_single_image(img_path, get_visual=True)
        visual = output["visual"]

        visualizer.vis_named_img("kps_out", visual[None], transpose=True, denormalize=False)

        for key in output.keys():
            print(key)

        time.sleep(1)
