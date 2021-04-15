# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.


import torch
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

from iPERCore.tools.human_digitalizer.bodynets import SMPL


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


class SmplLinker(object):
    def __init__(self,
                 smpl_model="assets/pretrains/smpl_model.pkl",
                 part_path="assets/pretrains/smpl_part_info.json",
                 device=torch.device("cuda:0")):

        self.smpl_model_path = smpl_model
        self.part_path = part_path

        self.mean_shape = torch.from_numpy(
            np.array([-0.00124704, 0.00200815, 0.01044902, 0.01385473, 0.01137672,
                      -0.01685408, 0.0201432, -0.00677187, 0.0050879, -0.0051118])).float()

        self.smpl = SMPL(model_path=self.smpl_model_path).to(device)

        with open(self.part_path, 'r') as f:
            # dict_keys(['00_head', '01_torso', '02_left_leg', '03_right_leg', '04_left_arm',
            #            '05_right_arm', '06_left_foot', '07_right_foot', '08_left_hand', '09_right_hand'])
            self.smpl_part_info = json.load(f)

        self.right_leg_verts_idx = np.array(self.smpl_part_info['03_right_leg']['vertex'])
        self.left_leg_verts_idx = np.array(self.smpl_part_info['04_left_arm']['vertex'])

        self.right_leg_inner_verts_idx = self.get_inner_verts_idx_of_leg(self.right_leg_verts_idx, inner_part_rate=0.3,
                                                                         right=True)
        self.left_leg_inner_verts_idx = self.get_inner_verts_idx_of_leg(self.left_leg_verts_idx, inner_part_rate=0.3,
                                                                        right=False)

    @torch.no_grad()
    def get_inner_verts_idx_of_leg(self, leg_verts_idx, inner_part_rate=0.3, right=True):
        """

        Args:
            leg_verts_idx (np.ndarray or list): (leg_verts_num,)
            inner_part_rate (float):
            right (bool): if True, x will be sorted from small to large (for right leg)

        Returns:
            innter_verts_idx (np.ndarray or list): (inner_verts_num,)

        """

        T_pose = create_360_degree_T_Pose_view_smpl(frame_num=1, ret_quat=False).reshape(1, 72)
        T_pose = torch.from_numpy(T_pose).float().cuda()
        mean_shape = self.mean_shape[None].cuda()

        verts, _, _ = self.smpl(mean_shape, T_pose, get_skin=True)
        # (6890, 3)
        verts = verts.detach().cpu().numpy()[0]
        # (leg_verts_num,)
        leg_verts_x = verts[leg_verts_idx][:, 0]

        inner_verts_num = int(leg_verts_idx.shape[0] * inner_part_rate)

        if right:
            inner_verts_idx_idx = np.argsort(leg_verts_x)[:inner_verts_num]
        else:
            inner_verts_idx_idx = np.argsort(leg_verts_x)[::-1][:inner_verts_num]

        inner_verts_idx = leg_verts_idx[inner_verts_idx_idx]
        return inner_verts_idx

    @staticmethod
    def find_nearest_vert(src_vert, tgt_vert_idx_list, tgt_verts):
        """

        Args:
            src_vert (list or np.ndarray or torch.tensor): (3,);
            tgt_vert_idx_list (list or np.ndarray): (N,) save verts idx;
            tgt_verts (np.ndarray or torch.tensor): (N, 3);

        Returns:
            [(vert_idx_0, dis_0), (vert_idx_1, dis_1)]

        """

        src_vert = src_vert.reshape(1, 3)
        # # (N, ), Euclidean Distance of two points
        # dis_array = ((src_vert - tgt_verts) ** 2).sum(axis=1)

        # (N, ), Distance only on y-axis
        dis_array = ((src_vert[:, 1:2] - tgt_verts[:, 1:2]) ** 2).sum(axis=1)

        dis0_idx = np.argsort(dis_array)[0]

        return tgt_vert_idx_list[dis0_idx], dis_array[dis0_idx]

    def link(self, cam, pose, shape, skirt_y, ret_tensor=True):
        """

        Args:
            cam (torch.cuda.tensor): (1, 3)
            pose (torch.cuda.tensor): (1, 72)
            shape (torch.cuda.tensor): (1, 10)
            skirt_y (float):
            ret_tensor (bool):

        Returns:
            from_verts_idx (np.ndarray or torch.tensor): (N,)
            to_verts_idx (np.ndarray or torch.tensor): (N,)

        """

        cam = cam.detach().cpu().numpy()[0]
        origin_verts, _, _ = self.smpl(shape, pose, get_skin=True)
        # (6890, 3)
        origin_verts = origin_verts[0].detach().cpu().numpy()

        # (left_leg_verts_num, 3)
        left_leg_verts = origin_verts[self.left_leg_verts_idx]
        # (right_leg_verts_num, 3)
        right_leg_verts = origin_verts[self.right_leg_verts_idx]

        from_vert_idx_list = []
        to_vert_idx_list = []
        for right_vert_idx in self.right_leg_inner_verts_idx:
            right_vert = origin_verts[right_vert_idx]
            vert_idx_0, dis_0 = self.find_nearest_vert(right_vert, self.left_leg_verts_idx, left_leg_verts)

            cam_right_vert_y = (right_vert[1] + cam[2]) * cam[0]
            if cam_right_vert_y <= skirt_y:
                from_vert_idx_list.append(right_vert_idx)
                to_vert_idx_list.append(vert_idx_0)

        for left_vert_idx in self.left_leg_inner_verts_idx:
            left_vert = origin_verts[left_vert_idx]
            vert_idx_0, dis_0 = self.find_nearest_vert(left_vert, self.right_leg_verts_idx, right_leg_verts)

            cam_left_vert_y = (left_vert[1] + cam[2]) * cam[0]
            if cam_left_vert_y <= skirt_y:
                from_vert_idx_list.append(left_vert_idx)
                to_vert_idx_list.append(vert_idx_0)

        from_verts_idx = np.array(from_vert_idx_list)
        to_verts_idx = np.array(to_vert_idx_list)

        if ret_tensor:
            from_verts_idx = torch.from_numpy(from_verts_idx).long().cuda()
            to_verts_idx = torch.from_numpy(to_verts_idx).long().cuda()

        return from_verts_idx, to_verts_idx
