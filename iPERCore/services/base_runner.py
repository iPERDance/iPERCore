import os
import numpy as np
import torch.backends.cudnn as cudnn
from scipy.spatial.transform import Rotation as R


def create_T_pose_novel_view_smpl(length=180):
    """

    Args:
        length (int):

    Returns:
        smpls (np.ndarray): (length, 85)
    """
    # cam + pose + shape = (3, 85)
    smpls = np.zeros((length, 85), dtype=np.float32)

    for i in range(length):
        r1 = R.from_rotvec([0, 0, 0])
        r2 = R.from_euler("xyz", [180, i * 2, 0], degrees=True)
        r = (r1 * r2).as_rotvec()

        smpls[i, 3:6] = r

    return smpls


def create_360_degree_global_rot(frame_num, ret_quat=False):
    """

    Args:
        frame_num (int): the number of frame;
        ret_quat (bool): if it is True, returns quat (N, num_joints, 4),
                         otherwise, returns rotvec (N, num_joints, 3).

    Returns:
        The global rotation.

        global_rot (np.ndarray): (N, num_joints, 4) or (N, num_joints, 3)

    """

    global_rot_euler_array = []
    delta = 360 / (frame_num - 1) if frame_num > 1 else 0

    for i in range(frame_num):
        # to be same with hmr version
        x = -np.pi
        y = delta * i / 180.0 * np.pi
        z = 0

        global_rot_euler_array.append([x, y, z])

    # (N, 3)
    global_rot_euler_array = np.array(global_rot_euler_array)

    if ret_quat:
        global_rot = R.from_euler(seq="xyz", angles=global_rot_euler_array, degrees=False).as_quat()
    else:
        global_rot = R.from_euler(seq="xyz", angles=global_rot_euler_array, degrees=False).as_rotvec()

    return global_rot


def add_hands_params_to_smpl(smpls, hands_param):
    """

    Args:
        smpls (np.ndarray): (length, 85)
        hands_param (np.ndarary): (length, 90) or (90,)

    Returns:
        smplhs (np.ndarray): (length, 156)

    """
    n_num1 = smpls.shape[0]

    if hands_param.ndim == 1:
        hands_param = np.tile(hands_param, reps=(n_num1, 1))

    cams = smpls[:, 0:3]
    pose = smpls[:, 3:-10]
    shape = smpls[:, -10:]

    smplhs = np.concatenate([cams, pose[:, 0:66], hands_param, shape], axis=1)

    return smplhs


def use_cudnn():
    # cudnn related setting
    cudnn.benchmark = True
    # cudnn.deterministic = False
    cudnn.deterministic = True
    cudnn.enabled = True




