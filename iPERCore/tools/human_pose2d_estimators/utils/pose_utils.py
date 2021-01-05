# original file comes from Copyright (c) 2018 algo, https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/pose.py
# this modified file Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import cv2
import numpy as np

from .one_euro_filter import OneEuroFilter


colors = {
    "pink": [197, 27, 125],  # L lower leg
    "Violet": [238, 130, 238],
    "DarkViolet": [148, 0, 211],
    "light_pink": [233, 163, 201],  # L upper leg
    "light_green": [161, 215, 106],  # L lower arm
    "green": [77, 146, 33],  # L upper arm
    "IndianRed": [205, 92, 92],
    "RosyBrown2": [238, 180, 180],
    "red": [215, 48, 39],  # head
    "light_red": [252, 146, 114],  # head
    "light_orange": [252, 141, 89],  # chest
    "DarkOrange2": [238, 118, 0],
    "purple": [118, 42, 131],  # R lower leg
    "BlueViolet": [138, 43, 226],
    "light_purple": [175, 141, 195],  # R upper
    "light_blue": [145, 191, 219],  # R lower arm
    "MediumSlateBlue": [123, 104, 238],
    "DarkSlateBlue": [72, 61, 139],
    "NavyBlue": [0, 0, 128],
    "LightSlateBlue": [132, 112, 255],
    "blue": [69, 117, 180],  # R upper arm
    "gray": [130, 130, 130],  #
    "YellowGreen": [154, 205, 50],
    "LightCoral": [240, 128, 128],
    "Aqua": [0, 255, 255],
    "chocolate": [210, 105, 30],
    "white": [255, 255, 255],  #
}


jcolors = [
    "light_red", "light_pink", "light_green", "red", "pink", "green",
    "light_orange", "light_purple", "light_blue", "DarkOrange2", "purple", "blue",
    "MediumSlateBlue", "YellowGreen", "LightCoral", "YellowGreen", "green", "LightSlateBlue", "MediumSlateBlue",
    "DarkSlateBlue", "DarkSlateBlue", "Violet", "BlueViolet", "NavyBlue", "RosyBrown2", "Aqua", "chocolate"
]


ecolors = {
    0: "IndianRed",
    1: "RosyBrown2",
    2: "light_pink",
    3: "pink",
    4: "Violet",
    5: "DarkViolet",
    6: "light_blue",
    7: "DarkSlateBlue",
    8: "LightSlateBlue",
    9: "NavyBlue",
    10: "MediumSlateBlue",
    11: "blue",
    12: "BlueViolet",
    13: "DarkSlateBlue",
    14: "purple",
    15: "Violet",
    16: "BlueViolet",
    17: "RosyBrown2",
    18: "light_green",
    19: "YellowGreen",
    20: "light_red",
    21: "light_pink",
    22: "light_green",
    23: "pink",
    24: "green",
    25: "chocolate",
    26: "Aqua"
}


class BasePose(object):

    @classmethod
    def get_bbox(cls, keypoints):
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.float32)
        found_kpt_id = 0
        for kpt_id in range(cls.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id, 0:2]
            found_kpt_id += 1

        ## (x0, y0, w, h)
        # bbox = cv2.boundingRect(found_keypoints)

        # (x0, y0, x1, y1)
        x0, y0 = np.min(found_keypoints, axis=0)
        x1, y1 = np.max(found_keypoints, axis=0)

        bbox = np.array([x0, y0, x1, y1], dtype=np.float32)
        return bbox

    @classmethod
    def get_similarity(cls, a, b, threshold=0.5):
        num_similar_kpt = 0
        for kpt_id in range(cls.num_kpts):
            if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
                distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
                area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
                similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * cls.vars[kpt_id]))
                if similarity > threshold:
                    num_similar_kpt += 1
        return num_similar_kpt

    @classmethod
    def track_poses(cls, previous_poses, current_poses, threshold=3, smooth=False):
        """
        Propagate poses ids from previous frame results. Id is propagated,
        if there are at least `threshold` similar keypoints between pose from previous frame and current.
        If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.

        Args:
            previous_poses: poses from previous frame with ids;
            current_poses: poses from current frame to assign ids;
            threshold: minimal number of similar keypoints between poses;
            smooth: smooth pose keypoints between frames.

        Returns:
            current_poses (list of BasePose): the current poses.
        """
        current_poses = sorted(current_poses, key=lambda pose: pose.confidence,
                               reverse=True)  # match confident poses first
        mask = np.ones(len(previous_poses), dtype=np.int32)
        for current_pose in current_poses:
            best_matched_id = None
            best_matched_pose_id = None
            best_matched_iou = 0
            for id, previous_pose in enumerate(previous_poses):
                if not mask[id]:
                    continue
                iou = cls.get_similarity(current_pose, previous_pose)
                if iou > best_matched_iou:
                    best_matched_iou = iou
                    best_matched_pose_id = previous_pose.id
                    best_matched_id = id
            if best_matched_iou >= threshold:
                mask[best_matched_id] = 0
            else:  # pose not similar to any previous
                best_matched_pose_id = None
            current_pose.update_id(best_matched_pose_id)

            if smooth:
                for kpt_id in range(cls.num_kpts):
                    if current_pose.keypoints[kpt_id, 0] == -1:
                        continue
                    # reuse filter if previous pose has valid filter
                    if (best_matched_pose_id is not None
                        and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
                        current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                    current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](
                        current_pose.keypoints[kpt_id, 0])
                    current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](
                        current_pose.keypoints[kpt_id, 1])
                current_pose.bbox = cls.get_bbox(current_pose.keypoints)

        return current_poses

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = self.last_id + 1
            self.last_id += 1

    def draw(self, img, radius=6):
        assert self.keypoints.shape == (self.num_kpts, 3)

        for part_id in range(len(self.BODY_PARTS_IDS_RENDER)):
            kpt_a_id = self.BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a, s_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), radius, colors[jcolors[kpt_a_id]], -1)
            kpt_b_id = self.BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b, s_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), radius, colors[jcolors[kpt_b_id]], -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), colors[ecolors[part_id]], radius // 2)


class OpenPoseBody25(BasePose):

    num_kpts = 25
    pose_entry_size = 27
    kpt_names = [
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee",
        "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe",
        "RSmallToe", "RHeel"
    ]

    BODY_PARTS_KPT_IDS = [
        (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6),
        (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13),
        (13, 14), (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
        (2, 17), (5, 18), (14, 19), (19, 20), (14, 21), (11, 22),
        (22, 23), (11, 24)
    ]

    BODY_PARTS_PAF_IDS = [
        (0, 1), (14, 15), (22, 23), (16, 17), (18, 19), (24, 25),
        (26, 27), (6, 7), (2, 3), (4, 5), (8, 9), (10, 11),
        (12, 13), (30, 31), (32, 33), (36, 37), (34, 35), (38, 39),
        (20, 21), (28, 29), (40, 41), (42, 43), (44, 45), (46, 47),
        (48, 49), (50, 51)
    ]

    BODY_PARTS_IDS_RENDER = [
        (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6),
        (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13),
        (13, 14), (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
        (14, 19), (19, 20), (14, 21), (11, 22),
        (22, 23), (11, 24)
    ]

    sigmas = np.array([
        .26,  .79,  .79, .72,  .62, .79, .72, .62,
        # TODO, additional mid-hip
        0.79,
        1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35,
        # TODO, additional 6 feets
        .35, .35,  .25, .35, .35,  .25
    ], dtype=np.float32) / 10.0

    vars = (sigmas * 2) ** 2
    last_id = -1

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = self.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(self.num_kpts)]


POSE_CLASS = {
    "BODY_25": OpenPoseBody25
}
