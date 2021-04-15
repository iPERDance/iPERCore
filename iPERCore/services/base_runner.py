# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import numpy as np
import torch.backends.cudnn as cudnn
from scipy.spatial.transform import Rotation as R

from iPERCore.tools.utils.filesio.cv_utils import load_parse, read_cv2_img, normalize_img


def create_T_pose_novel_view_smpl(length=180):
    """

    Args:
        length:
    Returns:

    """

    smpls = np.zeros((length, 85), dtype=np.float32)

    delta = 360 / (length - 1) if length > 1 else 0

    for i in range(length):
        y = delta * i
        r = R.from_euler("xyz", [180, y, 0], degrees=True).as_rotvec()

        smpls[i, 3:6] = r

    return smpls


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


def add_view_effect(smpls, view_dir):
    """

    Args:
        smpls (np.ndarray): (n, 85)
        view_dir (float):

    Returns:
        smpls (np.ndarray): (n, 85)
    """
    length = len(smpls)

    r = R.from_euler("xyz", [0, view_dir, 0], degrees=True)

    for i in range(length):
        orig_r = R.from_rotvec(smpls[i, 3:6])
        cur_r = (r * orig_r).as_rotvec()
        smpls[i, 3:6] = cur_r

    return smpls


def add_bullet_time_effect(smpls, img_paths, bt_list):
    """

    Args:
        smpls (np.ndarray): (n, 85);
        img_paths (list of str): (n,);
        bt_list (list of tuple): (number of Bullet Effects, 2=(frame_id, duration)), eg [(f_1, n_1), ..., (f_k, n_k)]

    Returns:
        smpls (np.ndarray): (n + n_1 + ... + n_k, 85)
        img_paths (list of str): (n + n_1 + ... + n_k)
    """

    original_length = len(smpls)

    # ignore the frame_id >= original_length
    valid_bt_list = []
    new_length = original_length
    for frame_id, bullet_duration in bt_list:
        if frame_id < original_length:
            new_length += bullet_duration
            valid_bt_list.append((frame_id, bullet_duration))

    effect_smpls = []
    effect_img_paths = []

    start_id = 0
    for frame_id, bullet_duration in valid_bt_list:
        novel_smpls = create_T_pose_novel_view_smpl(length=bullet_duration)
        novel_smpls[:, -10:] = smpls[frame_id, -10:]
        novel_smpls[:, 6:-10] = smpls[frame_id, 6:-10]
        novel_smpls[:, 0:3] = smpls[frame_id, 0:3]

        effect_smpls.append(smpls[start_id:frame_id])
        effect_smpls.append(novel_smpls)

        effect_img_paths.extend(img_paths[start_id:frame_id])
        effect_img_paths.extend(img_paths[frame_id:frame_id + 1] * bullet_duration)

        start_id = frame_id

    effect_smpls.append(smpls[start_id:original_length])
    effect_img_paths.extend(img_paths[start_id:original_length])

    effect_smpls = np.concatenate(effect_smpls, axis=0)

    return effect_smpls, effect_img_paths


def add_special_effect(smpls, img_paths, view_dir=None, bt_list=None):
    """

    Args:
        smpls:
        img_paths:
        view_dir:
        bt_list:

    Returns:

    """

    effect_smpls = smpls
    effect_img_paths = img_paths

    if view_dir is not None:
        effect_smpls = add_view_effect(smpls, view_dir)

    if bt_list is not None:
        effect_smpls, effect_img_paths = add_bullet_time_effect(effect_smpls, img_paths, bt_list)

    return effect_smpls, effect_img_paths


def get_src_info_for_inference(opt, vid_info):
    """

    Args:
        opt:
        vid_info (dict):

    Returns:

    """
    img_dir = vid_info["img_dir"]
    src_ids = vid_info["src_ids"]
    image_names = vid_info["images"]

    alpha_paths = vid_info["alpha_paths"]
    inpainted_paths = vid_info["inpainted_paths"]
    actual_bg_path = vid_info["actual_bg_path"]

    masks = []
    for i in src_ids:
        parse_path = alpha_paths[i]
        mask = load_parse(parse_path, opt.image_size)
        masks.append(mask)

    if actual_bg_path is not None:
        bg_img = read_cv2_img(actual_bg_path)
        bg_img = normalize_img(bg_img, image_size=opt.image_size, transpose=True)

    elif opt.use_inpaintor:
        bg_img = read_cv2_img(inpainted_paths[0])
        bg_img = normalize_img(bg_img, image_size=opt.image_size, transpose=True)

    else:
        bg_img = None

    src_info_for_inference = {
        "paths": [os.path.join(img_dir, image_names[i]) for i in src_ids],
        "smpls": vid_info["smpls"][src_ids],
        "offsets": vid_info["offsets"],
        "links": vid_info["links"],
        "masks": masks,
        "bg": bg_img
    }

    return src_info_for_inference


def get_src_info_for_swapper_inference(opt, vid_info_list):
    """

    Args:
        opt:
        vid_info_list (list of dict):

    Returns:
        src_info_for_inference
    """

    src_info_for_inference = {
        "paths": [],
        "src_paths": [],
        "smpls": [],
        "offsets": [],
        "links": [],
        "masks": [],
        "bg": [],
        "swap_parts": [],
        "num_source": []
    }

    merged_num_source = 0
    for i, vid_info in enumerate(vid_info_list):
        num_source = vid_info["num_source"]
        parts = vid_info["input_info"]["meta_input"]["parts"]

        src_info = get_src_info_for_inference(opt, vid_info)

        merged_num_source += num_source

        src_info_for_inference["src_paths"].extend(src_info["paths"])

        src_info_for_inference["num_source"].append(num_source)
        src_info_for_inference["paths"].append(src_info["paths"])
        src_info_for_inference["smpls"].append(src_info["smpls"])
        src_info_for_inference["masks"].append(src_info["masks"])
        src_info_for_inference["links"].append(src_info["links"])
        src_info_for_inference["offsets"].append(src_info["offsets"])
        src_info_for_inference["swap_parts"].append(parts)
        src_info_for_inference["bg"].append(src_info["bg"])

    return src_info_for_inference


# def get_src_info_for_swapper_inference(opt, vid_info_list, primary_ids=0):
#     """
#
#     Args:
#         opt:
#         vid_info_list (list of dict):
#         primary_ids (int):
#
#     Returns:
#         src_info_for_inference
#     """
#
#     src_info_for_inference = {
#         "paths": [],
#         "smpls": [],
#         "offsets": [],
#         "links": [],
#         "masks": [],
#         "bg": [],
#         "swap_parts": [],
#         "num_source": []
#     }
#
#     merged_num_source = 0
#     for i, vid_info in enumerate(vid_info_list):
#         num_source = vid_info["num_source"]
#         parts = vid_info["input_info"]["meta_input"]["parts"]
#
#         src_info = get_src_info_for_inference(opt, vid_info)
#
#         merged_num_source += num_source
#         links = []
#         offsets = []
#         swap_parts = []
#         for _ in range(num_source):
#             if src_info["links"] is None:
#                 _links = np.array([], dtype=np.int32)
#             else:
#                 _links = src_info["links"]
#
#             links.append(_links)
#             offsets.append(src_info["offsets"])
#             swap_parts.append(parts)
#
#         # set the background image of the primary source images.
#         if i == primary_ids:
#             src_info_for_inference["bg"] = src_info["bg"]
#
#         src_info_for_inference["num_source"].append(num_source)
#         src_info_for_inference["paths"].extend(src_info["paths"])
#         src_info_for_inference["smpls"].append(src_info["smpls"])
#         src_info_for_inference["masks"].append(src_info["masks"])
#         src_info_for_inference["links"].extend(links)
#         src_info_for_inference["offsets"].extend(offsets)
#         src_info_for_inference["swap_parts"].extend(swap_parts)
#
#         print(src_info["smpls"].shape)
#
#     # [(ns_1, 85), ..., (ns_k, 85)] -> (ns = ns_1 + ... + ns_k, 85)
#     src_info_for_inference["smpls"] = np.concatenate(src_info_for_inference["smpls"], axis=0)
#
#     # [(ns_1, 1, h, w), ..., (ns_k, 1, h, w)] -> (ns = ns_1 + ... + ns_k, 1, h, w)
#     src_info_for_inference["masks"] = np.concatenate(src_info_for_inference["masks"], axis=0)
#
#     # [(num_verts, 3), ..., (num_verts, 3)]
#     src_info_for_inference["offsets"] = np.stack(src_info_for_inference["offsets"], axis=0)
#
#     # [(num_verts, 3), ..., (num_verts, 3)]
#     src_info_for_inference["links"] = np.stack(src_info_for_inference["links"], axis=0)
#
#     # update number source
#     opt.num_source = merged_num_source
#     print(f"update the number of sources {src_info_for_inference['num_source']} = {merged_num_source}")
#
#     print(src_info_for_inference["smpls"].shape)
#     print(src_info_for_inference["masks"].shape)
#     print(src_info_for_inference["offsets"].shape)
#     print(src_info_for_inference["links"].shape)
#
#     return src_info_for_inference
