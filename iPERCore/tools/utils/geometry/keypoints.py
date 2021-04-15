# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import abc
import numpy as np

from iPERCore.tools.utils.signals.smooth import temporal_filter_invalid_kps, get_smooth_params


VALID_DATASET = [
    "OpenPose-Body-25",
    "OpenPose-Body-18",
    "Coco-Body-17",
    "CocoWhole-Body-23",
    "CocoWhole-Body-Hand-65",
    "CocoWhole-Body-Hand-Face-133"
]


POSE2D_OUT_KEYS = [
    "pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d",
    "pose_keypoints_3d", "face_keypoints_3d", "hand_left_keypoints_3d", "hand_right_keypoints_3d"
]


class KeypointFormater(metaclass=abc.ABCMeta):

    @staticmethod
    def temporal_smooth_keypoints(stack_keypoints):
        """

        Args:
            stack_keypoints (dict): the stacked keypoints information, it contains,
                --pose_keypoints_2d (np.ndarray): (n, 25 * 3) = (n, 25 * (x,y,score));

        Returns:
            ts_keypoints (dict): the temporal smoothed keypoints information, it contains,
                --pose_keypoints_2d (np.ndarray): (n, 25 * 3) = (n, 25 * (x,y,score))
        """

        ts_keypoints = dict()
        for key_name, val in stack_keypoints.items():
            num_frames = val.shape[0]
            if key_name in POSE2D_OUT_KEYS and num_frames > 10:
                sm_kps = temporal_filter_invalid_kps(val)
                sm_kps = get_smooth_params(sm_kps)
                ts_keypoints[key_name] = sm_kps
            else:
                ts_keypoints[key_name] = val

        return ts_keypoints

    @abc.abstractmethod
    def format_keypoints(self, keypoints, im_shape):
        """

        Format the keypoints, normalize the range from [0, height/width] to [0, 224] for simplify.
        If im_shape is None, then keypoints must be [-1, 1].

        Args:
            keypoints (dict): the keypoints information, and it contains,
                --pose_keypoints_2d (np.ndarray or list): (25 * 3) = (25 * (x,y,score))
                --face_keypoints_2d (np.ndarray or list): ()
                --hand_left_keypoints_2d (np.ndarray or list):
                --hand_right_keypoints_2d (np.ndarray or list):(25 * 3) = (25 * (x,y,score))

            im_shape (tuple or list or None): im_shape[0] = height, im_shape[1] = width

        Returns:
            kps (np.ndarray): (75, 3) the re-normalized keypoints.
        """
        pass

    @abc.abstractmethod
    def format_stacked_keypoints(self, ids, keypoints, im_shape):
        """
        Format the stacked keypoints, normalize the range from [0, height/width] to [0, 224] for simplify.

        Args:
            ids (int): the index.
            keypoints (dict): the keypoints information, it contains,
                --pose_keypoints_2d (np.ndarray): (n, num_joints * 3) = (n, num_joints * (x,y,score))

            im_shape (tuple or list): im_shape[0] = height, im_shape[1] = width

        Returns:
            kps (np.ndarray): (75, 3) the re-normalized keypoints.
        """
        pass

    @abc.abstractmethod
    def stack_keypoints(self, keypoints_list_or_dict):
        """
        Stack all list of keypoints dict in t-axis.

        Args:
            keypoints_list_or_dict (list of dict): [keypoints_1, keypoints_2, ..., keypoints_n]. Each keypoints_i contains,
                --pose_keypoints_2d (np.ndarray or list): (num_joints * 3) = (num_joints * (x,y,score))
                --face_keypoints_2d (np.ndarray or list): ()
                --hand_left_keypoints_2d (np.ndarray or list):
                --hand_right_keypoints_2d (np.ndarray or list):

        Returns:
            stacked_keypoints (dict): the stacked keypoints, it contains,
                --pose_keypoints_2d (np.ndarray): (n, num_joints * 3) = (n, num_joints * (x,y,score))
                --face_keypoints_2d (np.ndarray or list): ()
                --hand_left_keypoints_2d (np.ndarray or list):
                --hand_right_keypoints_2d (np.ndarray or list):
        """
        pass

    @abc.abstractmethod
    def mapper_to_smpl(self, num_smpl_joints):
        pass


class OpenPoseBody25KeypointFormater(KeypointFormater):

    NUM_JOINTS = 25
    JOINT_TYPE = "OpenPose-Body-25"
    JOINT_NAMES = [
        "Nose",          # 0
        "Neck",          # 1
        "RShoulder",     # 2
        "RElbow",        # 3
        "RWrist",        # 4
        "LShoulder",     # 5
        "LElbow",        # 6
        "LWrist",        # 7
        "MidHip",        # 8
        "RHip",          # 9
        "RKnee",         # 10
        "RAnkle",        # 11
        "LHip",          # 12
        "LKnee",         # 13
        "LAnkle",        # 14
        "REye",          # 15
        "LEye",          # 16
        "REar",          # 17
        "LEar",          # 18
        "LBigToe",       # 19
        "LSmallToe",     # 20
        "LHeel",         # 21
        "RBigToe",       # 22
        "RSmallToe",     # 23
        "RHeel",         # 24
    ]

    THIS_NAME_TO_SMPL_45 = {
        "Nose": 24,         "Neck": 12,         "RShoulder": 17,
        "RElbow": 19,       "RWrist": 21,       "LShoulder": 16,
        "LElbow": 18,       "LWrist": 20,       "MidHip": 0,
        "RHip": 2,          "RKnee": 5,         "RAnkle": 8,
        "LHip": 1,          "LKnee": 4,         "LAnkle": 7,
        "REye": 25,         "LEye": 26,         "REar": 27,
        "LEar": 28,         "LBigToe": 29,      "LSmallToe": 30,
        "LHeel": 31,        "RBigToe": 32,      "RSmallToe": 33,
        "RHeel": 34
    }

    def __init__(self, num_smpl_joints=45, ignore_joints=("Neck", "RHip", "LHip")):
        # ignore_joints=("Neck", "RHip", "LHip")
        self.mapper, self.ignore_ids, self.ignore_joints, self.num_smpl_joints = self.mapper_to_smpl(
            num_smpl_joints,  )

    def mapper_to_smpl(self, num_smpl_joints, ignore_joints=("Neck", "RHip", "LHip")):
        self.ignore_joints = ignore_joints
        self.mapper = [self.THIS_NAME_TO_SMPL_45[name] for name in self.JOINT_NAMES]
        self.ignore_ids = [self.THIS_NAME_TO_SMPL_45[name] for name in ignore_joints
                           if name in self.THIS_NAME_TO_SMPL_45]
        self.num_smpl_joints = num_smpl_joints

        return self.mapper, self.ignore_ids, self.ignore_joints, self.num_smpl_joints

    def format_keypoints(self, keypoints, im_shape=None):
        """

        Format the keypoints, normalize the range from [0, height/width] to [0, 224] for simplify.
        If im_shape is None, then keypoints must be [-1, 1].

        Args:
            keypoints (dict): the keypoints information, and it contains,
                --pose_keypoints_2d (np.ndarray or list): (25 * 3) = (25 * (x,y,score))
                --face_keypoints_2d (np.ndarray or list): ()
                --hand_left_keypoints_2d (np.ndarray or list):
                --hand_right_keypoints_2d (np.ndarray or list):(25 * 3) = (25 * (x,y,score))

            im_shape (tuple or list or None): im_shape[0] = height, im_shape[1] = width

        Returns:
            to_smpl_kps (np.ndarray): (self.num_smpl_joints, 3) the re-normalized keypoints.
        """
        kps = np.reshape(np.array(keypoints["pose_keypoints_2d"], copy=True, dtype=np.float32), (-1, 3))

        if im_shape is None:
            # (kps + 1) / 2 * 224
            kps[:, 0:2] = (kps[:, 0:2] + 1) * 112
        else:
            height, width = im_shape
            kps[:, 0] = kps[:, 0] / width * 224
            kps[:, 1] = kps[:, 1] / height * 224

        to_smpl_kps = np.zeros((self.num_smpl_joints, 3), dtype=np.float32)
        to_smpl_kps[self.mapper] = kps
        to_smpl_kps[self.ignore_ids] = 0

        return to_smpl_kps

    def format_stacked_keypoints(self, ids, keypoints, im_shape):
        """
        Format the keypoints, normalize the range from [0, height/width] to [0, 224] for simplify.

        Args:
            ids (int): the index.
            keypoints (dict): the keypoints information, it contains,
                --pose_keypoints_2d (np.ndarray): (n, 25 * 3) = (n, 25 * (x,y,score))

            im_shape (tuple or list): im_shape[0] = height, im_shape[1] = width

        Returns:
            kps (np.ndarray): (self.num_smpl_joints, 3) the re-normalized keypoints.
        """

        keypoints_info = {
            "pose_keypoints_2d": keypoints["pose_keypoints_2d"][ids]
        }

        kps = self.format_keypoints(keypoints_info, im_shape)
        return kps

    def stack_keypoints(self, keypoints_list):
        """
        Stack all list of keypoints dict in t-axis.

        Args:
            keypoints_list (dict):
                --pose_keypoints_2d (list of dict): [keypoints_info_1, keypoints_info_2, ..., keypoints_info_n],
                    each keypoints_info_i is dictionary, and it contains the following information:
                        --pose_keypoints_2d: (25 * 3,) = (25, xyz);
                --joint_type (str): OpenPose-Body-25;
                --im_shape(tuple): (height, width)

        Returns:
            stack_keypoints (dict): the stacked keypoints, it contains,
                --pose_keypoints_2d (np.ndarray): (n, 25 * 3) = (n, 25 * (x,y,score))
        """

        pose_keypoints_2d = []
        length = len(keypoints_list)

        for i in range(length):
            pose_keypoints_2d.append(keypoints_list[i]["pose_keypoints_2d"])

        pose_keypoints_2d = np.array(pose_keypoints_2d, dtype=np.float32)

        stack_keypoints = {
            "pose_keypoints_2d": pose_keypoints_2d
        }

        return stack_keypoints


class CocoWholeBody23KeypointFormater(KeypointFormater):

    NUM_JOINTS = 23
    JOINT_TYPE = "CocoWhole-Body-23"
    JOINT_NAMES = [
        # 23 CocoWhole-Body joints (in the order provided by CocoWhole-Body)
        "Nose",       # 0
        "LEye",       # 1
        "REye",       # 2
        "LEar",       # 3
        "REar",       # 4
        "LShoulder",  # 5
        "RShoulder",  # 6
        "LElbow",     # 7
        "RElbow",     # 8
        "LWrist",     # 9
        "RWrist",     # 10
        "LHip",       # 11
        "RHip",       # 12
        "LKnee",      # 13
        "RKnee",      # 14
        "LAnkle",     # 15
        "RAnkle",     # 16
        "LBigToe",    # 17
        "LSmallToe",  # 18
        "LHeel",      # 19
        "RBigToe",    # 20
        "RSmallToe",  # 21
        "RHeel",      # 22
    ]

    THIS_NAME_TO_SMPL_45 = {
        "Nose": 24,                     "RShoulder": 17,
        "RElbow": 19, "RWrist": 21,     "LShoulder": 16,
        "LElbow": 18, "LWrist": 20,
        "RHip": 2,    "RKnee": 5,       "RAnkle": 8,
        "LHip": 1,    "LKnee": 4,       "LAnkle": 7,
        "REye": 25,   "LEye": 26,       "REar": 27,
        "LEar": 28,   "LBigToe": 29,    "LSmallToe": 30,
        "LHeel": 31,  "RBigToe": 32,    "RSmallToe": 33,
        "RHeel": 34,
    }

    def __init__(self, num_smpl_joints=45, ignore_joints=("RHip", "LHip")):
        # ignore_joints=("RHip", "LHip")
        self.mapper, self.ignore_ids, self.ignore_joints, self.num_smpl_joints = self.mapper_to_smpl(
            num_smpl_joints, ignore_joints)

    def mapper_to_smpl(self, num_smpl_joints, ignore_joints=("RHip", "LHip")):
        self.ignore_joints = ignore_joints
        self.mapper = [self.THIS_NAME_TO_SMPL_45[name] for name in self.JOINT_NAMES]
        self.ignore_ids = [self.THIS_NAME_TO_SMPL_45[name] for name in ignore_joints
                           if name in self.THIS_NAME_TO_SMPL_45]
        self.num_smpl_joints = num_smpl_joints

        return self.mapper, self.ignore_ids, self.ignore_joints, self.num_smpl_joints

    def format_keypoints(self, keypoints, im_shape=None):
        """

        Format the keypoints, normalize the range from [0, height/width] to [0, 224] for simplify.
        If im_shape is None, then keypoints must be [-1, 1].

        Args:
            keypoints (dict): the keypoints information, and it contains,
                --pose_keypoints_2d (np.ndarray or list): (23 * 3) = (23 * (x,y,score))
                --face_keypoints_2d (np.ndarray or list): ()
                --hand_left_keypoints_2d (np.ndarray or list):
                --hand_right_keypoints_2d (np.ndarray or list):(23 * 3) = (23 * (x,y,score))

            im_shape (tuple or list or None): im_shape[0] = height, im_shape[1] = width

        Returns:
            to_smpl_kps (np.ndarray): (self.num_smpl_joints, 3) the re-normalized keypoints.
        """
        kps = np.reshape(np.array(keypoints["pose_keypoints_2d"], copy=True, dtype=np.float32), (-1, 3))

        if im_shape is None:
            # (kps + 1) / 2 * 224
            kps[:, 0:2] = (kps[:, 0:2] + 1) * 112
        else:
            height, width = im_shape
            kps[:, 0] = kps[:, 0] / width * 224
            kps[:, 1] = kps[:, 1] / height * 224

        to_smpl_kps = np.zeros((self.num_smpl_joints, 3), dtype=np.float32)
        to_smpl_kps[self.mapper] = kps
        to_smpl_kps[self.ignore_ids] = 0

        return to_smpl_kps

    def format_stacked_keypoints(self, ids, keypoints, im_shape):
        """
        Format the keypoints, normalize the range from [0, height/width] to [0, 224] for simplify.

        Args:
            ids (int): the index.
            keypoints (dict): the keypoints information, it contains,
                --pose_keypoints_2d (np.ndarray): (n, 23 * 3) = (n, 23 * (x,y,score))

            im_shape (tuple or list): im_shape[0] = height, im_shape[1] = width

        Returns:
            kps (np.ndarray): (45, 3) the re-normalized keypoints.
        """

        keypoints_info = {
            "pose_keypoints_2d": keypoints["pose_keypoints_2d"][ids]
        }

        kps = self.format_keypoints(keypoints_info, im_shape)
        return kps

    def stack_keypoints(self, keypoints_list):
        """
        Stack all list of keypoints dict in t-axis.

        Args:
            keypoints_list (List[Dict]): list of keypoints dictionary information, and each dict contains,
                --pose_keypoints_2d (np.ndarray or list): (23, 3) is in the range of [0, height/width];
                --face_keypoints_2d (np.ndarray or list): (68, 3) is in the range of [0, height/width];
                --hand_left_keypoints_2d (np.ndarray or list): (21, 3) is in the range of [0, height/width];
                --hand_right_keypoints_2d (np.ndarray or list): (21, 3) is in the range of [0, height/width].

        Returns:
            stack_keypoints (dict): the stacked keypoints, it contains,
                --pose_keypoints_2d (np.ndarray): (n, 25 * 3) = (n, 25 * (x,y,score))
        """

        pose_keypoints_2d = []
        length = len(keypoints_list)

        for i in range(length):
            pose_keypoints_2d.append(keypoints_list[i]["pose_keypoints_2d"])

        pose_keypoints_2d = np.array(pose_keypoints_2d, dtype=np.float32)

        stack_keypoints = {
            "pose_keypoints_2d": pose_keypoints_2d
        }

        return stack_keypoints


class HalpeBody26KeypointFormater(KeypointFormater):

    def __init__(self):
        self.joint_type = "Halpe-Body-26"
        self.num_joints = 26

    def format_keypoints(self, keypoints, im_shape=None):
        """

        Format the keypoints, normalize the range from [0, height/width] to [0, 224] for simplify.
        If im_shape is None, then keypoints must be [-1, 1].

        Args:
            keypoints (dict): the keypoints information, and it contains,
                --pose_keypoints_2d (np.ndarray or list): (26 * 3) = (26 * (x,y,score))
                --face_keypoints_2d (np.ndarray or list): ()
                --hand_left_keypoints_2d (np.ndarray or list):
                --hand_right_keypoints_2d (np.ndarray or list):(26 * 3) = (26 * (x,y,score))

            im_shape (tuple or list or None): im_shape[0] = height, im_shape[1] = width

        Returns:
            kps (np.ndarray): (75, 3) the re-normalized keypoints.
        """
        kps = np.reshape(np.array(keypoints["pose_keypoints_2d"], copy=True, dtype=np.float32), (-1, 3))

        if im_shape is None:
            # (kps + 1) / 2 * 224
            kps[:, 0:2] = (kps[:, 0:2] + 1) * 112
        else:
            height, width = im_shape
            kps[:, 0] = kps[:, 0] / width * 224
            kps[:, 1] = kps[:, 1] / height * 224

        kps = np.concatenate([np.zeros((25 + 24, 3), dtype=np.float32), kps], axis=0)

        return kps

    def format_stacked_keypoints(self, ids, keypoints, im_shape):
        """
        Format the keypoints, normalize the range from [0, height/width] to [0, 224] for simplify.

        Args:
            ids (int): the index.
            keypoints (dict): the keypoints information, it contains,
                --pose_keypoints_2d (np.ndarray): (n, 26 * 3) = (n, 26 * (x,y,score))

            im_shape (tuple or list): im_shape[0] = height, im_shape[1] = width

        Returns:
            kps (np.ndarray): (75, 3) the re-normalized keypoints.
        """

        keypoints_info = {
            "pose_keypoints_2d": keypoints["pose_keypoints_2d"][ids]
        }

        kps = self.format_keypoints(keypoints_info, im_shape)
        return kps

    def stack_keypoints(self, keypoints_list_or_dict):
        """
        Stack all list of keypoints dict in t-axis.

        Args:
            keypoints_list_or_dict (list of dict): [keypoints_1, keypoints_2, ..., keypoints_n].
                Each keypoints_i contains,
                    --pose_keypoints_2d (np.ndarray or list): (26 * 3) = (26 * (x,y,score))
                    --face_keypoints_2d (np.ndarray or list): ()
                    --hand_left_keypoints_2d (np.ndarray or list):
                    --hand_right_keypoints_2d (np.ndarray or list):

        Returns:
            stack_keypoints (dict): the stacked keypoints, it contains,
                --pose_keypoints_2d (np.ndarray): (n, 26 * 3) = (n, 25 * (x,y,score))
        """

        if isinstance(keypoints_list_or_dict, dict):
            stack_keypoints = keypoints_list_or_dict

        else:
            pose_keypoints_2d = []
            length = len(keypoints_list_or_dict)

            for i in range(length):
                pose_keypoints_2d.append(keypoints_list_or_dict[i]["pose_keypoints_2d"])

            pose_keypoints_2d = np.array(pose_keypoints_2d, dtype=np.float32)

            stack_keypoints = {
                "pose_keypoints_2d": pose_keypoints_2d
            }

        return stack_keypoints


KEYPOINTS_FORMATER = {
    "CocoWhole-Body-23": CocoWholeBody23KeypointFormater,
    "OpenPose-Body-25": OpenPoseBody25KeypointFormater,
    "Halpe-Body-26": HalpeBody26KeypointFormater
}
