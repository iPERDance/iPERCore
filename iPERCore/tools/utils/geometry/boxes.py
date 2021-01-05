# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import numpy as np
import torch


@torch.no_grad()
def cal_mask_bbox(head_mask, factor=1.25):
    """
    Args:
        head_mask (np.ndarray): (N, 1, 256, 256).
        factor (float): the factor to enlarge the bbox of head.

    Returns:
        bbox (np.ndarray.int32): (N, 4), hear, 4 = (x1, x2, y1, y2)

    """
    bs, _, height, width = head_mask.shape

    bbox = np.zeros((bs, 4), dtype=np.int32)
    valid = np.ones((bs,), dtype=np.float32)

    for i in range(bs):
        mask = head_mask[i, 0]
        ys, xs = np.where(mask != 0)

        if len(ys) == 0:
            valid[i] = 0.0
            bbox[i, 0] = 0
            bbox[i, 1] = width
            bbox[i, 2] = 0
            bbox[i, 3] = height
            continue

        lt_y = np.min(ys)   # left top of Y
        lt_x = np.min(xs)   # left top of X

        rt_y = np.max(ys)   # right top of Y
        rt_x = np.max(xs)   # right top of X

        h = rt_y - lt_y     # height of head
        w = rt_x - lt_x     # width of head

        cy = (lt_y + rt_y) // 2    # (center of y)
        cx = (lt_x + rt_x) // 2    # (center of x)

        _h = h * factor
        _w = w * factor

        _lt_y = max(0, int(cy - _h / 2))
        _lt_x = max(0, int(cx - _w / 2))

        _rt_y = min(height, int(cy + _h / 2))
        _rt_x = min(width, int(cx + _w / 2))

        if (_lt_x == _rt_x) or (_lt_y == _rt_y):
            valid[i] = 0.0
            bbox[i, 0] = 0
            bbox[i, 1] = width
            bbox[i, 2] = 0
            bbox[i, 3] = height
        else:
            bbox[i, 0] = _lt_x
            bbox[i, 1] = _rt_x
            bbox[i, 2] = _lt_y
            bbox[i, 3] = _rt_y

    return bbox, valid


@torch.no_grad()
def cal_head_bbox(kps, image_size):
    """
    Args:
        kps (torch.Tensor): (N, 19, 2)
        image_size (int):

    Returns:
        bbox (torch.Tensor): (N, 4)
    """
    NECK_IDS = 12  # in cocoplus

    kps = (kps + 1) / 2.0

    necks = kps[:, NECK_IDS, 0]
    zeros = torch.zeros_like(necks)
    ones = torch.ones_like(necks)

    # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
    min_x, _ = torch.min(kps[:, NECK_IDS:, 0] - 0.05, dim=1)
    min_x = torch.max(min_x, zeros)

    max_x, _ = torch.max(kps[:, NECK_IDS:, 0] + 0.05, dim=1)
    max_x = torch.min(max_x, ones)

    # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
    min_y, _ = torch.min(kps[:, NECK_IDS:, 1] - 0.05, dim=1)
    min_y = torch.max(min_y, zeros)

    max_y, _ = torch.max(kps[:, NECK_IDS:, 1], dim=1)
    max_y = torch.min(max_y, ones)

    min_x = (min_x * image_size).long()  # (T,)
    max_x = (max_x * image_size).long()  # (T,)
    min_y = (min_y * image_size).long()  # (T,)
    max_y = (max_y * image_size).long()  # (T,)

    rects = torch.stack((min_x, max_x, min_y, max_y), dim=1)
    return rects


@torch.no_grad()
def cal_body_bbox(kps, image_size, factor=1.25):
    """
    Args:
        kps (torch.cuda.FloatTensor): (N, 19, 2)
        image_size (int):
        factor (float):

    Returns:
        bbox: (N, 4)
    """
    bs = kps.shape[0]
    kps = (kps + 1) / 2.0
    zeros = torch.zeros((bs,), device=kps.device)
    ones = torch.ones((bs,), device=kps.device)

    min_x, _ = torch.min(kps[:, :, 0], dim=1)
    max_x, _ = torch.max(kps[:, :, 0], dim=1)
    middle_x = (min_x + max_x) / 2
    width = (max_x - min_x) * factor
    min_x = torch.max(zeros, middle_x - width / 2)
    max_x = torch.min(ones, middle_x + width / 2)

    min_y, _ = torch.min(kps[:, :, 1], dim=1)
    max_y, _ = torch.max(kps[:, :, 1], dim=1)
    middle_y = (min_y + max_y) / 2
    height = (max_y - min_y) * factor
    min_y = torch.max(zeros, middle_y - height / 2)
    max_y = torch.min(ones, middle_y + height / 2)

    min_x = (min_x * image_size).long()  # (T,)
    max_x = (max_x * image_size).long()  # (T,)
    min_y = (min_y * image_size).long()  # (T,)
    max_y = (max_y * image_size).long()  # (T,)

    # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
    bboxs = torch.stack((min_x, max_x, min_y, max_y), dim=1)

    return bboxs


@torch.no_grad()
def cal_head_bbox_by_mask(head_mask, factor=1.25):
    """
    Args:
        head_mask (torch.cuda.FloatTensor): (N, 1, 256, 256).
        factor (float): the factor to enlarge the bbox of head.

    Returns:
        bbox (np.ndarray.int32): (N, 4), hear, 4 = (left_top_x, left_top_y, right_top_x, right_top_y)

    """
    bs, _, height, width = head_mask.shape

    bbox = torch.zeros((bs, 4), dtype=torch.long)

    for i in range(bs):
        mask = head_mask[i, 0]
        coors = (mask == 1).nonzero(as_tuple=False)
        if len(coors) == 0:
            bbox[i, 0] = 0
            bbox[i, 1] = 0
            bbox[i, 2] = 0
            bbox[i, 3] = 0
        else:
            ys = coors[:, 0]
            xs = coors[:, 1]
            min_y = ys.min()  # left top of Y
            min_x = xs.min()  # left top of X

            max_y = ys.max()  # right top of Y
            max_x = xs.max()  # right top of X

            h = max_y - min_y  # height of head
            w = max_x - min_x  # width of head

            cy = (min_y + max_y) // 2  # (center of y)
            cx = (min_x + max_x) // 2  # (center of x)

            _h = h * factor
            _w = w * factor

            _lt_y = max(0, int(cy - _h / 2))
            _lt_x = max(0, int(cx - _w / 2))

            _rt_y = min(height, int(cy + _h / 2))
            _rt_x = min(width, int(cx + _w / 2))

            bbox[i, 0] = _lt_x
            bbox[i, 1] = _rt_x
            bbox[i, 2] = _lt_y
            bbox[i, 3] = _rt_y

    return bbox
