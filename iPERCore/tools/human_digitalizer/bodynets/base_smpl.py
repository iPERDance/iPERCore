# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch.nn as nn
import numpy as np


def batch_orth_proj_idrot(X, camera):
    """
    X is N x num_points x 3
    camera is N x 3
    same as applying orth_proj_idrot to each N
    """

    # TODO check X dim size.

    # X_trans is (N, num_points, 2)
    X_trans = X[:, :, :2] + camera[:, None, 1:]
    return camera[:, None, 0:1] * X_trans


class BaseSMPL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def link(self, verts, linked_ids):
        """

        Args:
            verts (torch.Tensor): (N, 6890, 3)
            linked_ids (np.ndarray or torch.tensor): (N, number of verts, 2 = (from_vert_ids, to_verts_ids);

        Returns:
            linked_verts (torch.Tensor): (N, 6890, 3)
        """

        bs = verts.shape[0]

        linked_verts = verts.clone()

        if len(linked_ids.shape) == 2:
            linked_verts[:, linked_ids[:, 0]] = verts[:, linked_ids[:, 1]]
        else:
            for i in range(bs):
                has_linked = (linked_ids[i, :, 2] == 1)
                linked_verts[i, linked_ids[i, has_linked, 0]] = verts[i, linked_ids[i, has_linked, 1]]

        return linked_verts

    def split(self, theta) -> dict:
        """
        Args:
            theta: (N, 3 + 72 + 10)
        Returns:
            detail_info (dict): it contains the following information,
                --cam (torch.Tensor): (N, 3)
                --pose (torch.Tensor): (N, 72)
                --shape (torch.Tensor): (N, 10)
                --theta (torch.Tensor): (N, 85)
        """

        cam = theta[:, 0:3]
        pose = theta[:, 3:-10].contiguous()
        shape = theta[:, -10:].contiguous()

        detail_info = {
            "cam": cam,
            "pose": pose,
            "shape": shape,
            "theta": theta
        }

        return detail_info

    def skinning(self, theta, offsets=0, links_ids=None) -> dict:
        """
        Args:
            theta: (N, 3 + 72 + 10);
            offsets (torch.Tensor) : (N, nv, 3) or 0;
            links_ids (None or np.ndarray or torch.tensor): (nv, 2) or (bs, nv, 2)
        Returns:

        """

        cam = theta[:, 0:3]
        pose = theta[:, 3:-10].contiguous()
        shape = theta[:, -10:].contiguous()
        verts, j3d, rs = self.forward(beta=shape, theta=pose, offsets=offsets, links_ids=links_ids, get_skin=True)

        detail_info = {
            "cam": cam,
            "pose": pose,
            "shape": shape,
            "verts": verts,
            "theta": theta
        }

        return detail_info

    def get_details(self, theta, offsets=0, links_ids=None) -> dict:
        """
            calc verts, joint2d, joint3d, Rotation matrix
        Args:
            theta (torch.Tensor): (N, 85) = (N , 3 + 72 + 10)
            offsets (torch.Tensor) : (N, nv, 3) or 0
            links_ids (None or np.ndarray or torch.tensor): (nv, 2) or (bs, nv, 2)
        Returns:
            detail_info (dict): the details information of smpl, including
                --theta (torch.Tensor): (N, 85),
                --cam (torch.Tensor):   (N, 3),
                --pose (torch.Tensor):  (N, 72),
                --shape (torch.Tensor): (N, 10),
                --verts (torch.Tensor): (N, 6890, 3),
                --j2d (torch.Tensor):   (N, 19, 2),
                --j3d (torch.Tensor):   (N, 19, 3)

        """

        cam = theta[:, 0:3]
        pose = theta[:, 3:-10].contiguous()
        shape = theta[:, -10:].contiguous()
        verts, j3d, rs = self.forward(beta=shape, theta=pose, offsets=offsets, links_ids=links_ids, get_skin=True)
        j2d = batch_orth_proj_idrot(j3d, cam)

        detail_info = {
            "theta": theta,
            "cam": cam,
            "pose": pose,
            "shape": shape,
            "verts": verts,
            "j2d": j2d,
            "j3d": j3d
        }

        return detail_info

