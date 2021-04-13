# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import numpy as np
from scipy.spatial.transform import Rotation as R


def create_360_degree_T_Pose_view_smpl(frame_num, ret_quat=True):
    """

    Args:
        frame_num (int): number of frame
        ret_quat (bool): if True,  return quat (N, 24, 4)
                         if False, return rotvec (N, 24, 3)

    Returns:
       quat (np.ndarray): if ret_quat is True, (N, 24, 4)
       recvec (np.ndarray): if ret_quat is False, (N, 24, 3)
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
    # (N, 1, 4)
    global_quat_array = R.from_euler(seq='xyz', angles=global_rot_euler_array, degrees=False).as_quat()[:, None]

    # (N, 3)
    body_rot_vec_array = np.zeros((frame_num * 23, 3))
    # (N, 23, 4)
    body_quat_array = R.from_rotvec(body_rot_vec_array).as_quat().reshape(frame_num, 23, 4)

    # (N, 24, 4)
    quat_pose_array = np.concatenate([global_quat_array, body_quat_array], axis=1)

    if ret_quat:
        ret_pose_array = quat_pose_array
    else:
        ret_pose_array = R.from_quat(quat_pose_array.reshape(frame_num * 24, 4)).as_rotvec().reshape(frame_num, 24, 3)

    return ret_pose_array

