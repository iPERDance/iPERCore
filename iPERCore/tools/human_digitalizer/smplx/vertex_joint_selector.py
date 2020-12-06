# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn

from .utils import to_tensor


class VertexJointSelector(nn.Module):

    def __init__(self, vertex_ids=None,
                 use_hands=True,
                 use_feet_keypoints=True, **kwargs):
        super(VertexJointSelector, self).__init__()

        extra_joints_idxs = []

        face_keyp_idxs = np.array([
            vertex_ids['nose'],
            vertex_ids['reye'],
            vertex_ids['leye'],
            vertex_ids['rear'],
            vertex_ids['lear']], dtype=np.int64)

        extra_joints_idxs = np.concatenate([extra_joints_idxs,
                                            face_keyp_idxs])

        if use_feet_keypoints:
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'],
                                       vertex_ids['LSmallToe'],
                                       vertex_ids['LHeel'],
                                       vertex_ids['RBigToe'],
                                       vertex_ids['RSmallToe'],
                                       vertex_ids['RHeel']], dtype=np.int32)

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, feet_keyp_idxs])

        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, tips_idxs])

        self.register_buffer('extra_joints_idxs',
                             to_tensor(extra_joints_idxs, dtype=torch.long))

    def forward(self, vertices, joints):
        """
        Selects the extra joints from vertices, and then concatenates the original joints with the extra joints.

        1. If `self.use_feet_keypoints == False` and `self.use_hands == False`, then the extra joints is empty:
            1.1 if the model is `SMPL`, then it returns joints, and the output shape is (batch_size, 24, 3);

            1.2 if the model is `SMPL-H`, then it returns joints, and the output shape is (batch_size, 52, 3);

            #TODO 1.3 if the model is `SMPL-X`, then it returns joints, and the output shape is xxxxxxxxxxxxxxxxxxx;

        2. If `self.use_feet_keypoints == True` and `self.use_hands == False`,

           2.1 if the model is `SMPL`, then it returns [joints, `nose`, `reye`, `leye`, `rear`, `lear`,
           `LBigToe`, `LSmallToe`, `LHeel`, `RBigToe`, `RSmallToe`, `RHeel`],
           and the output shape is (batch_size, 24 + 5 + 6 = 35, 3);

           2.2 if the model is `SMPL-H`, then it returns [joints, `nose`, `reye`, `leye`, `rear`, `lear`,
           `LBigToe`, `LSmallToe`, `LHeel`, `RBigToe`, `RSmallToe`, `RHeel`],
           and the output shape is (batch_size, 52 + 5 + 6 = 63, 3).

           #TODO 2.3 if the model is `SMPL-X`

        3. If `self.use_feet_keypoints == True` and `self.use_hands == True`, then the output is [joints, face, hands]:
            3.1 if the model is `SMPL-H`, then it returns [joints, `nose`, `reye`, `leye`, `rear`, `lear`,
            `LBigToe`, `LSmallToe`, `LHeel`, `RBigToe`, `RSmallToe`, `RHeel`,
            `thumb`, `index`, `middle`, `ring`, `pinky`], and the shape is (batch_size, 24 + 5 + 6 + 10 = 45, 3)

            3.2 if the model is `SMPL-H`, then it returns [joints, `nose`, `reye`, `leye`, `rear`, `lear`,
            `LBigToe`, `LSmallToe`, `LHeel`, `RBigToe`, `RSmallToe`, `RHeel`,
            `thumb`, `index`, `middle`, `ring`, `pinky`], and the shape is (batch_size, 52 + 5 + 6 + 10 = 73, 3)

            #TODO 3.3 if the model is `SMPL-X`

        Args:
            vertices (torch.tensor): (batch_size, 6890, 3) or (batch_size, 10475, 3).
            joints (torch.tensor): (batch_size, number of joints, 3).

        Returns:

        """
        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([joints, extra_joints], dim=1)

        return joints
