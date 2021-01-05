# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import cv2
import numpy as np
import os
import os.path as osp
from typing import Union, List, Tuple
import platform
import shutil

from iPERCore.tools.utils.filesio.persistence import mkdir
from iPERCore.tools.utils.multimedia import is_image_file, is_video_file, video2frames


def format_imgs_dir(src_path, imgs_dir):

    if not osp.exists(src_path):
        raise FileNotFoundError(f"{src_path} does not exist. Please provide a existing path.")

    if is_image_file(src_path):
        # if `src_path` is a image path, the the image will be copied to `output_dir/frames`.
        imgs_dir = mkdir(imgs_dir)
        dst_path = osp.join(imgs_dir, osp.split(src_path)[-1])

        # create a symbolic link
        if src_path != dst_path and not osp.exists(dst_path):

            # TODO, handle the Privilege Error (WinError 1314) of os.symlink in Windows? Need help.
            if platform.system().lower() == "windows":
                shutil.copy(src_path, dst_path)
            else:
                os.symlink(osp.abspath(src_path), osp.abspath(dst_path))

    elif is_video_file(src_path):
        imgs_dir = mkdir(imgs_dir)
        # * if `src_path` is a video path, then use `ffmpeg` to extract the frames into `output_dir/frames`.
        video2frames(src_path, imgs_dir)

    elif osp.isdir(src_path):
        # * if `src_path` is a directory that contains multiple images, then these images will
        # be copied to `output_dir/frames`.
        if src_path != imgs_dir and not osp.exists(imgs_dir):

            # TODO, handle the Privilege Error (WinError 1314) of os.symlink in Windows? Need help.
            if platform.system().lower() == "windows":
                if osp.exists(imgs_dir):
                    shutil.rmtree(imgs_dir)
                shutil.copytree(src_path, imgs_dir)
            else:
                os.symlink(osp.abspath(src_path), osp.abspath(imgs_dir))

    else:
        raise ValueError("imgs_dir {} is not image path, video path and image directory.".format(imgs_dir))

    return imgs_dir


def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * np.float(scale_factor))).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def get_approximate_square_crop_boxes(orig_shape, active_bbox):
    """
    Args:
        orig_shape:
        active_bbox (list): [min_x, max_x, min_y, max_y];

    Returns:

    """

    orig_h, orig_w = orig_shape

    min_x, min_y, max_x, max_y = active_bbox

    box_h = int(max_y - min_y)
    box_w = int(max_x - min_x)

    # print("orig = {}, active_bbox = {}, boxes = {}".format(orig_shape, active_bbox, (box_h, box_w)))
    if box_h > box_w:
        pad_size = box_h - box_w
        pad = pad_size // 2
        if pad_size % 2 == 0:
            pad_1, pad_2 = pad, pad
        else:
            pad_1, pad_2 = pad, pad + 1

        min_x = max(0, min_x - pad_1)
        max_x = min(orig_w, max_x + pad_2)

    else:
        pad_size = box_w - box_h
        pad = pad_size // 2
        if pad_size % 2 == 0:
            pad_1, pad_2 = pad, pad
        else:
            pad_1, pad_2 = pad, pad + 1

        min_y = max(0, min_y - pad_1)
        max_y = min(orig_h, max_y + pad_2)

    return min_x, min_y, max_x, max_y


def update_active_boxes(cur_boxes, active_boxes=None):
    """

    Args:
        cur_boxes:
        active_boxes:

    Returns:

    """
    if active_boxes is None:
        active_boxes = cur_boxes
    else:
        active_boxes[0] = min(active_boxes[0], cur_boxes[0])
        active_boxes[1] = min(active_boxes[1], cur_boxes[1])
        active_boxes[2] = max(active_boxes[2], cur_boxes[2])
        active_boxes[3] = max(active_boxes[3], cur_boxes[3])

    return active_boxes


def fmt_active_boxes(active_boxes_XYXY, orig_shape, factor):
    boxes = enlarge_boxes(active_boxes_XYXY, orig_shape, factor)
    return pad_boxes(boxes, orig_shape)


def enlarge_boxes(active_boxes_XYXY,
                  orig_shape: Union[List[int], Tuple[int]],
                  factor: float = 1.125):

    x0, y0, x1, y1 = active_boxes_XYXY
    height, width = orig_shape

    h = y1 - y0  # height

    ctr_x = (x0 + x1) // 2  # (center of x)
    ctr_y = (y0 + y1) // 2  # (center of y)

    _h = h * factor

    _y0 = max(0, int(ctr_y - _h / 2))
    _y1 = min(height, int(ctr_y + _h / 2))
    __h = _y1 - _y0

    _x0 = max(0, int(ctr_x - __h / 2))
    _x1 = min(width, int(ctr_x + __h / 2))

    return _x0, _y0, _x1, _y1


def pad_boxes(boxes_XYXY, orig_shape):
    orig_h, orig_w = orig_shape

    x0, y0, x1, y1 = boxes_XYXY

    box_h = int(x1 - x0)
    box_w = int(y1 - y0)

    if box_h > box_w:
        pad_size = box_h - box_w
        pad = pad_size // 2
        if pad_size % 2 == 0:
            pad_1, pad_2 = pad, pad
        else:
            pad_1, pad_2 = pad, pad + 1

        x0 = max(0, x0 - pad_1)
        x1 = min(orig_w, x1 + pad_2)

    else:
        pad_size = box_w - box_h
        pad = pad_size // 2
        if pad_size % 2 == 0:
            pad_1, pad_2 = pad, pad
        else:
            pad_1, pad_2 = pad, pad + 1

        y0 = max(0, y0 - pad_1)
        y1 = min(orig_h, y1 + pad_2)

    return x0, y0, x1, y1


def process_crop_img(orig_img, active_bbox, image_size):
    """
    Args:
        orig_img (np.ndarray):
        active_bbox (4,) : [x0, y0, x1, y1]
        image_size (int):

    Returns:
        image_info (dict): the information of processed image,
            `image` (np.ndarray): the crop and resized image, the shape is (image_size, image_size, 3);
            `orig_shape` (tuple): the shape of the original image;
            `active_bbox` (tuple or list): the active bbox [min_x, max_x, min_y, max_y];
            `factor`: the fact to enlarge the active bbox;
            `crop_bbox`: the cropped bbox [min_x, max_x, min_y, max_y];
            `pad_bbox`: the padded bbox [pad_left_x, pad_right_x, pad_top_y, pad_bottom_y],
    """

    x0, y0, x1, y1 = get_approximate_square_crop_boxes(orig_img.shape[0:2], active_bbox)
    crop_img = orig_img[y0: y1, x0: x1, :]
    crop_h, crop_w = crop_img.shape[0:2]

    start_pt = np.array([x0, y0], dtype=np.float32)

    pad_size = max(crop_h, crop_w) - min(crop_h, crop_w)
    pad = pad_size // 2
    if pad_size % 2 == 0:
        pad_1, pad_2 = pad, pad
    else:
        pad_1, pad_2 = pad, pad + 1

    # 1161 485 1080 1080 (1080, 1080, 3) (595, 0, 3) 595 297 298
    # print(x0, y0, x1, y1, orig_img.shape, crop_img.shape, pad_size, pad_1, pad_2)
    if crop_h < crop_w:
        crop_img = np.pad(
            array=crop_img,
            pad_width=((pad_1, pad_2), (0, 0), (0, 0)),
            mode="edge"
        )
        start_pt -= np.array([0, pad_1], dtype=np.float32)

    elif crop_h > crop_w:
        crop_img = np.pad(
            array=crop_img,
            pad_width=((0, 0), (pad_1, pad_2), (0, 0)),
            mode="edge"
        )
        start_pt -= np.array([pad_1, 0], dtype=np.float32)

    pad_crop_size = crop_img.shape[0]

    scale = image_size / pad_crop_size
    start_pt *= scale

    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=np.float32)
    center *= scale
    center -= start_pt

    proc_img = cv2.resize(crop_img, (image_size, image_size))

    return {
        # return original too with info.
        "image": proc_img,
        "im_shape": orig_img.shape[0:2],
        "center": center,
        "scale": scale,
        "start_pt": start_pt,
    }


def crop_resize_boxes(boxes, scale, start_pt):
    """
        crop and resize the boxes in the original image coordinates into cropped image coordinates.
    Args:
        boxes (list):
        scale:
        start_pt:

    Returns:
        new_boxes (list):
    """

    x0, y0, x1, y1 = np.copy(boxes)
    x0 = x0 * scale - start_pt[0]
    y0 = y0 * scale - start_pt[1]
    x1 = x1 * scale - start_pt[0]
    y1 = y1 * scale - start_pt[1]

    new_boxes = [x0, y0, x1, y1]

    return new_boxes


def crop_resize_kps(keypoints, scale, start_pt):
    """
        crop and resize the keypoints in the original image coordinates into cropped image coordinates.
    Args:
        keypoints (dict):
        scale:
        start_pt:

    Returns:
        new_kps (dict):
    """

    new_kps = dict()

    for part, kps in keypoints.items():
        if len(kps) > 0:
            renorm_kps = kps.copy()
            renorm_kps[:, 0:2] = renorm_kps[:, 0:2] * scale - start_pt[np.newaxis]
            new_kps[part] = renorm_kps
        else:
            new_kps[part] = kps

    return new_kps


def norm_kps(keypoints, orig_shape, norm_type="01=>-1+1"):
    """

    Args:
        keypoints (dict):
        orig_shape (tuple): (height, width)
        norm_type (str): if norm_type is `01`, then normalize the kps into [0, 1],
                         if norm_type is `-1+1`, then normalize the kps into [-1, 1].
                         if norm_type us `hw`, then normalize the kps into [0, h/w].

    Returns:
        new_kps (dict):
    """

    new_kps = dict()

    height, width = orig_shape
    orig_size = np.array([width, height], dtype=np.float32)[np.newaxis]

    src_type, tgt_type = norm_type.split("=>")

    for part, kps in keypoints.items():
        if len(kps) > 0:
            renorm_kps = kps.copy()

            if src_type == "01" and tgt_type == "hw":
                renorm_kps[:, 0:2] *= orig_size
            elif src_type == "01" and tgt_type == "-1+1":
                renorm_kps[:, 0:2] *= 2
                renorm_kps[:, 0:2] -= 1
            elif src_type == "-1+1" and tgt_type == "01":
                renorm_kps[:, 0:2] += 1
                renorm_kps[:, 0:2] /= 2
            elif src_type == "-1+1" and tgt_type == "hw":
                renorm_kps[:, 0:2] += 1
                renorm_kps[:, 0:2] /= 2
                renorm_kps[:, 0:2] *= orig_size
            elif src_type == "hw" and tgt_type == "01":
                renorm_kps[:, 0:2] /= orig_size
            elif src_type == "hw" and tgt_type == "-1+1":
                renorm_kps[:, 0:2] /= orig_size
                renorm_kps[:, 0:2] *= 2
                renorm_kps[:, 0:2] -= 1

            new_kps[part] = renorm_kps
        else:
            new_kps[part] = kps

    return new_kps


def crop_func(src_path, out_path, crop_size, orig_shape, fmt_active_boxes, boxes_XYXY, keypoints):
    """

    Args:
        src_path:
        out_path:
        crop_size:
        orig_shape:
        fmt_active_boxes:
        boxes_XYXY:
        keypoints:

    Returns:
        crop_boxes (tuple):
        crop_kps (dict):
    """

    height, width = orig_shape
    orig_img = cv2.imread(src_path)

    if orig_img.shape[0:2] != orig_shape:
        orig_img = cv2.resize(orig_img, (width, height))

    crop_info = process_crop_img(orig_img, fmt_active_boxes, crop_size)

    cv2.imwrite(out_path, crop_info["image"])

    scale = crop_info["scale"]
    start_pt = crop_info["start_pt"]

    if boxes_XYXY is not None:
        orig_boxes = boxes_XYXY
        crop_boxes = crop_resize_boxes(orig_boxes, scale, start_pt)
    else:
        crop_boxes = None

    if keypoints is not None:
        orig_kps = keypoints
        crop_kps = crop_resize_kps(orig_kps, scale, start_pt)
    else:
        crop_kps = None

    return crop_boxes, crop_kps

