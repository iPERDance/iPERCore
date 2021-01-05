# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import numpy as np
import torch
import scipy.signal as signal
from scipy.ndimage import filters
from scipy import interpolate
from scipy.spatial.transform.rotation import Rotation as R

from iPERCore.tools.utils.geometry import rotations


VALID_FILTERS = ["low-pass", "median"]


def temporal_filter_invalid_kps(seq_kps):
    """

    Args:
        seq_kps:

    Returns:

    """
    length, num_joints, _ = seq_kps.shape

    for i in range(num_joints):
        kps_i = seq_kps[:, i]

        invalid = kps_i[:, 2] == 0

        valid = ~invalid

        valid_ids = np.where(valid)[0]

        f = interpolate.interp1d(valid_ids, kps_i[valid_ids], axis=0, kind="linear",
                                 fill_value="extrapolate", assume_sorted=True)

        invalid_ids = np.where(invalid)[0]
        kps_new = f(invalid_ids)

        seq_kps[invalid_ids, i] = kps_new

        # print(i, len(invalid_ids), length)

    return seq_kps


def fist_order_low_pass_filter(signal, alpha=0.7):
    """

    Y(n) = alpha * Y(n-1) + (1 - alpha) * X(n).

    Args:
        signal (np.ndarray or torch.Tensor): (n, c);
        alpha (float): Y(n) = alpha * Y(n-1) + (1 - alpha) * X(n)

    Returns:
        sm_signal (np.ndarray or torch.Tensor): (n, c)
    """

    if isinstance(signal, np.ndarray):
        sm_signal = np.copy(signal)
    else:
        sm_signal = signal.clone()

    n = signal.shape[0]

    for i in range(1, n):
        sm_signal[i] = alpha * sm_signal[i - 1] + (1 - alpha) * signal[i]

    return sm_signal


def get_smooth_params(sig, n=5, fc=300):
    """
    Low-pass filters.

    Args:
        sig (np.ndarray): (length, feature dimension);
        n (int): the number of low-pass order;
        fc (float): the factor to control the degree of smoothness. The smaller of fc, the more smoother.

    Returns:
        smooth_sig (np.ndarray): (length, feature dimension).

    """

    fs = 2208
    w = fc / (fs / 2)

    b, a = signal.butter(n, w, 'low')
    smooth_sig = signal.filtfilt(b, a, sig.T).T
    return smooth_sig


def mean_filter(sig, kernel_size):
    """
    Mean-Filters.

    Args:
        sig (np.ndarray): (n1, n2, n3, ...., nk);
        kernel_size (tuple): the kernel size, (s1, s2, s3, ..., nk).

    Returns:
        filtered_sig (np.ndarray): (length, feature dimension).

    """

    filtered_sig = filters.median_filter(sig, size=kernel_size, mode="nearest")

    return filtered_sig


def pose2d_distance(kps1, kps2):
    """

    Args:
        kps1 (np.ndarray): (length, num_joints_1, 2)
        kps2 (np.ndarray): (length, num_joints_2, 2)

    Returns:

    """

    n, num_joints = kps1.shape[0:2]
    assert n == kps2.shape[0]

    # (length, num_joints_1, 2) -> (length, num_joints_1, 1, 2) -> (length, num_joints_1, num_joints_2, 2)
    kps1 = np.tile(kps1[:, :, np.newaxis, :], reps=(1, 1, num_joints, 1))

    # (length, num_joints_2, 2) -> (length, 1, num_joints_2, 2) -> (length, num_joints_1, num_joints_2, 2)
    kps2 = np.tile(kps2[:, np.newaxis, :, :], reps=(1, num_joints, 1, 1))

    # (length, num_joints_1, num_joints_2)
    dist = np.sum((kps1 - kps2) ** 2, axis=-1)

    return dist


def pose2d_temporal_filter(keypoints, window_size, mode, **kwargs):
    """
    Temporal filter the keypoints. It mainly focuses on fixing the case that the coordinates of the keypoints are
    estimated successfully, while the it fails on the right-left orders. Here, we try to deal with it by following
    strategies:
        1. temporal smooth the keypoints, get the filtered_kps, and the smooth mode can be `mean`, `low-pass`;
        2. calculate the distance between the original keypoints and the filtered keypoints, and find the nearest
        neighbour joints of the original keypoints to the filtered_kps;
        3. permutate the keypoints based on the nearest neighbour indexes.

    Args:
        keypoints (np.ndarray): the original keypoints, (length, 2) or (length, 3);
        window_size (int): the size of temporal window;
        mode (str): the mode name of filters. Currently, it support,
            `median` for median filters;
            `low-pass` for low-pass filter;
        **kwargs: the other parameters, such as
            --fc (float): the smooth factor for low-pass filters.

    Returns:
        sm_keypoints (np.ndarray): the smoothed keypoints, (length, 2) or (length, 3).

    """

    global VALID_FILTERS

    if mode == "median":
        filtered_kps = mean_filter(keypoints, kernel_size=(window_size, 1, 1))
    elif mode == "low-pass":
        filtered_kps = get_smooth_params(keypoints, n=window_size, fc=kwargs["fc"])
    else:
        raise ValueError(f"{mode} is not valid mode. Currently, it only support {VALID_FILTERS}.")

    his_kps = filtered_kps[:, :, 0:2]
    query_kps = keypoints[:, :, 0:2]

    # (length, num_joints, num_joints)
    dist = pose2d_distance(query_kps, his_kps)
    nn_ids = np.argmin(dist, axis=2)

    length, num_joints, c = keypoints.shape

    ids = np.arange(length) * num_joints
    ids = ids[:, np.newaxis] + nn_ids

    # TODO, make it more readable and use numpy advanced indexing.
    sm_keypoints = keypoints.reshape(-1, c)[ids, :].reshape(-1, num_joints, c)

    return sm_keypoints


def temporal_smooth_smpls(ref_smpls, pose_fc=300, cam_fc=100):
    """

    Args:
        ref_smpls (np.ndarray): (length, 72)
        pose_fc:
        cam_fc:

    Returns:
        ref_smpls (np.ndarray): (length, 72)
    """
    ref_rotvec = ref_smpls[:, 3:-10]

    n = ref_rotvec.shape[0]
    ref_rotvec = ref_rotvec.reshape((-1, 3))
    ref_rotmat = R.from_rotvec(ref_rotvec).as_matrix()
    ref_rotmat = torch.from_numpy(ref_rotmat)
    ref_rot6d = rotations.rotmat_to_rot6d(ref_rotmat)
    ref_rot6d = ref_rot6d.numpy()
    ref_rot6d = ref_rot6d.reshape((n, -1))
    ref_rot6d = get_smooth_params(ref_rot6d, fc=pose_fc)
    ref_rot6d = ref_rot6d.reshape((-1, 6))
    ref_rotmat = rotations.rot6d_to_rotmat(torch.from_numpy(ref_rot6d)).numpy()
    ref_rotvec = R.from_matrix(ref_rotmat).as_rotvec()
    ref_smpls[:, 3:-10] = ref_rotvec.reshape((n, -1))

    ref_smpls[:, 0:3] = get_smooth_params(ref_smpls[:, 0:3], fc=cam_fc)

    return ref_smpls


def pose_temporal_smooth(init_pose_rotvec, opt_pose_rotvec, threshold: float = 10):
    """

    Args:
        init_pose_rotvec (torch.Tensor): (n, 72);
        opt_pose_rotvec (torch.Tensor): (n, 72);
        threshold (float):

    Returns:
        opt_pose_rotvec (torch.Tensor): (n, 72).
    """

    n = opt_pose_rotvec.shape[0]

    init_pose = rotations.rotvec_to_rot6d(init_pose_rotvec.reshape(-1, 3)).reshape(n, -1)
    opt_pose = rotations.rotvec_to_rot6d(opt_pose_rotvec.reshape(-1, 3)).reshape(n, -1)

    # sm_init_pose = smooth_poses.fist_order_low_pass_filter(init_pose, alpha=0.7)
    # sm_opt_pose = smooth_poses.fist_order_low_pass_filter(opt_pose, alpha=0.7)

    diff = torch.sum(torch.abs(init_pose - opt_pose), dim=1)
    abnormal_ids = (diff > threshold)

    opt_pose_rotvec[abnormal_ids] = init_pose_rotvec[abnormal_ids]

    return opt_pose_rotvec



