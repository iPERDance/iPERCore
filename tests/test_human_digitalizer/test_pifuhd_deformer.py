# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.


import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

from iPERCore.tools.utils.geometry.mesh import load_obj, save_to_obj
from iPERCore.tools.utils.filesio.persistence import load_json_file
from iPERCore.tools.human_digitalizer.bodynets import SMPL
from iPERCore.tools.human_digitalizer.pifuhd.pifuhd_runner import PifuHDRunner

from iPERCore.tools.human_digitalizer.deformers.utils import create_360_degree_T_Pose_view_smpl


class PifuHD2SMPLRegister(object):
    def __init__(self, total_iters=600, smpl_model="assets/checkpoints/pose3d/smpl_model.pkl",
                 facial_path="assets/checkpoints/pose3d/front_facial.json",
                 part_path="./assets/checkpoints/pose3d/smpl_part_info.json",
                 fit_pifuhd_path="assets/checkpoints/pose3d/mapper_for_fit_pifuhd.txt",
                 device=torch.device("cuda:0")):

        # config file path
        self.smpl_pkl_path = smpl_model
        self.facial_info_path = facial_path
        self.part_path = part_path
        self.uv_map_path = fit_pifuhd_path

        # training parameters
        self.iters_num = 300
        # Weight for shape reconstruction loss
        # self.w_recons = 1000
        # # Weight for mesh edge loss
        # self.w_edge = 300
        # # Weight for mesh laplacian smoothing
        # self.w_laplacian = 30

        self.w_recons = 2000
        # Weight for mesh edge loss
        self.w_edge = 500
        # Weight for mesh laplacian smoothing
        self.w_laplacian = 50

        self.w_face = 300
        self.w_foot = 300
        self.w_hand = 300

        # self.lr = 1e-2
        self.lr = 2e-2
        self.batch_size = 6890

        # self.w_deform_reg = 100
        self.w_deform_reg = 10
        self.w_y_reg = 10
        self.T_pose_reg = 1000

        self.w_deform_pose_reg = 20

        self.smpl_obj_info = load_obj(self.uv_map_path)

        # dict_keys(["0_head", "1_body", "2_left_arm", "3_right_arm", "4_left_leg",
        #            "5_right_leg", "6_left_foot", "7_right_foot", "8_left_hand", "9_right_hand"])
        part_info = load_json_file(self.part_path)

        # dict_keys(["vertex", "face"])
        facial_info = load_json_file(self.facial_info_path)

        # important vertex
        self.smpl_verts_weights = np.ones((6890,))
        self.smpl_verts_weights[np.array(facial_info["vertex"])] *= self.w_face
        # self.smpl_verts_weights[np.array(part_info["8_left_hand"]["vertex"] + part_info["9_right_hand"]["vertex"])] *= self.w_hand
        # self.smpl_verts_weights[np.array(part_info["6_left_foot"]["vertex"] + part_info["7_right_foot"]["vertex"])] *= self.w_foot
        self.smpl_verts_weights[
            np.array(part_info["08_left_hand"]["vertex"] + part_info["09_right_hand"]["vertex"])] *= self.w_hand
        self.smpl_verts_weights[
            np.array(part_info["06_left_foot"]["vertex"] + part_info["07_right_foot"]["vertex"])] *= self.w_foot

        # (6890, )
        self.smpl_verts_weights = torch.from_numpy(self.smpl_verts_weights).float().to(device)
        # (13776, 3)
        self.smpl_faces = torch.from_numpy(self.smpl_obj_info["faces"]).long().to(device)

        # pose weights: 1 for changeable pose and 0 for else
        self.smpl_pose_weights = np.ones((24,))
        # self.smpl_pose_weights[[7, 8, 20, 21, 22, 23]] = 0
        self.smpl_pose_weights[[7, 8, 20, 21, 22, 23, 12, 15]] = 0
        # self.smpl_pose_weights[[12, 15]] = 0

        # self.smpl_pose_weights = np.zeros((24,))
        # self.smpl_pose_weights[[9]] = 1

        self.smpl_pose_weights = torch.from_numpy(self.smpl_pose_weights).float().to(device)
        # (96, )
        self.smpl_pose_weights = self.smpl_pose_weights[:, None].repeat(1, 4).reshape(-1)

        # T-pose
        self.T_pose_array = create_360_degree_T_Pose_view_smpl(frame_num=5, ret_quat=True).reshape(-1, 96)
        self.T_pose_array = torch.from_numpy(self.T_pose_array).float().to(device)

        # model
        self.smpl = SMPL(self.smpl_pkl_path).to(device)

        self.device = device

    def load_pifuhd_verts(self, pifuhd_obj_path):
        """

        Args:
            pifuhd_obj_path (str): the obj path of pifuHD.

        Returns:
            pifu_verts (torch.cuda.Tensor): (N1, 3), here N1 is the number of vertices.

        """

        # load pifuhd verts
        obj_info = load_obj(pifuhd_obj_path)
        # (N1, 3)
        pifu_verts = obj_info["vertices"]
        # match smpl verts
        pifu_verts[:, 1:] *= -1

        # prepare torch data
        # (N1, 3)
        pifu_verts = torch.from_numpy(pifu_verts).float().to(self.device)

        return pifu_verts

    def initialize(self, cam, pose, shape, pifuhd_obj_path):
        """
        Converts all the inputs into torch.cuda.Tensor.

        Args:
            cam (np.ndarray):  (1, 3)
            pose (np.ndarray): (1, 72)
            shape (np.ndarray): (1, 10)
            pifuhd_obj_path (str): the obj path of pifuHD.

        Returns:
            cam (torch.cuda.Tensor): (1, 3);
            pose (torch.cuda.Tensor): (1, 72);
            shape (torch.cuda.Tensor): (1, 10);
            pifu_verts (torch.cuda.Tensor): (N1, 3), (N1, 3), here N1 is the number of vertices of pifuHD.
        """

        pose = torch.from_numpy(pose).float().to(self.device)
        shape = torch.from_numpy(shape).float().to(self.device)
        cam = torch.from_numpy(cam).float().to(self.device)

        pifu_verts = self.load_pifuhd_verts(pifuhd_obj_path)

        return cam, pose, shape, pifu_verts

    def fit_pifuhd(self, cam, pose, shape, pifuhd_obj_path, verbose=False):

        """

        Args:
            cam (np.ndarray or torch.Tensor): (1, 3);
            pose (np.ndarray or torch.Tensor): (1, 72);
            shape (np.ndarray or torch.Tensor): (1, 10);
            pifuhd_obj_path (str):
            verbose (bool):

        Returns:
            new_pose (np.ndarray): (1, 72)
            deform_verts (np.ndarray): (6890, 3)
        """

        """
        load pifuhd verts
        """
        obj_info = load_obj(pifuhd_obj_path)
        # (N1, 3)
        pifu_verts = obj_info["vertices"]
        # match smpl verts
        pifu_verts[:, 1:] *= -1

        """
        prepare torch data
        """
        quat_pose = R.from_rotvec(pose.reshape(-1, 3)).as_quat()
        # (1, 96)
        quat_pose = torch.from_numpy(quat_pose.reshape(1, 96)).float().to(self.device)

        cam = torch.from_numpy(cam).float().to(self.device)
        shape = torch.from_numpy(shape).float().to(self.device)

        # (N1, 3)
        pifu_verts = torch.from_numpy(pifu_verts).float().to(self.device)
        # (6890, 3)
        deform_verts = torch.zeros(6890, 3).float().to(self.device)
        deform_verts.requires_grad = True
        deform_quat_pose = torch.zeros(1, 96).float().to(self.device)
        deform_quat_pose.requires_grad = True

        optimizer = torch.optim.Adam([deform_verts, deform_quat_pose], lr=self.lr, betas=(0.9, 0.999), weight_decay=0)

        bar = range(1, self.iters_num + 1)
        if verbose:
            bar = tqdm(bar)

        """
        optimization
        """
        new_verts = None
        for i in bar:
            optimizer.zero_grad()
            # (batch_size, 3)
            sampled_pifu_verts = self.sample_verts(pifu_verts, sample_num=self.batch_size)

            weighted_deform_quat_pose = deform_quat_pose * self.smpl_pose_weights[None]
            # weighted_deform_quat_pose = deform_quat_pose
            new_quat_pose = self.update_quat_pose(quat_pose, weighted_deform_quat_pose)

            new_verts, _, _ = self.smpl(shape, new_quat_pose, deform_verts[None], get_skin=True)
            new_verts[:, :, :2] += cam[:, None, 1:]
            new_verts *= cam[:, None, 0:1]

            new_T_verts, _, _ = self.smpl(shape, self.T_pose_array[0, None], deform_verts[None], get_skin=True)

            new_smpl_mesh = Meshes(verts=new_verts, faces=self.smpl_faces[None])

            recons_loss = self.compute_recons_loss(new_smpl_mesh.verts_packed(), self.smpl_faces,
                                                   sampled_pifu_verts) * self.w_recons
            edge_loss = mesh_edge_loss(new_smpl_mesh) * self.w_edge
            laplacian_loss = mesh_laplacian_smoothing(new_smpl_mesh, method="uniform") * self.w_laplacian

            deform_reg_loss = self.compute_deform_reg_loss(deform_verts, self.smpl_verts_weights) * self.w_deform_reg

            y_reg_loss = self.compute_y_reg_loss(deform_verts) * self.w_y_reg
            deform_pose_reg_loss = (deform_quat_pose ** 2).mean() * self.w_deform_pose_reg

            # T_pose_reg_loss = (new_T_verts[:, :, 2].mean() ** 2) * 10000
            T_pose_reg_loss = (new_T_verts[:, :, 2] ** 2).var() * self.T_pose_reg

            """ total loss """
            loss = recons_loss + edge_loss + laplacian_loss + deform_reg_loss + y_reg_loss + T_pose_reg_loss + deform_pose_reg_loss

            # Optimization step
            loss.backward()
            optimizer.step()

        """
        final
        """
        new_quat_pose = self.update_quat_pose(quat_pose, deform_quat_pose).detach().cpu().numpy().reshape(-1, 24, 4)
        # (1, 96)
        new_pose = R.from_quat(new_quat_pose.reshape(-1, 4)).as_rotvec()
        new_pose = new_pose.reshape(1, 72)

        return new_pose, deform_verts, new_verts[0]

    def export_mesh(self, vertices, out_obj_path):
        """

        Args:
            vertices (np.ndarray): (6890, 3)
            out_obj_path:

        Returns:
            None
        """

        save_to_obj(
            path=out_obj_path,
            verts=vertices, faces=self.smpl_obj_info["faces"],
            vts=self.smpl_obj_info["vts"], vns=self.smpl_obj_info["vns"],
            faces_vts=self.smpl_obj_info["faces_vts"], faces_vns=self.smpl_obj_info["faces_vns"]
        )

    @staticmethod
    def update_quat_pose(quat_pose, deform_quat_pose):
        """

        Args:
            quat_pose (torch.Tensor): (N, 96)
            deform_quat_pose (torch.Tensor): (N, 96)

        Returns:
            new_quant_pose (torch.Tensor): (N, 96)

        """

        new_quat_pose = quat_pose + deform_quat_pose
        new_quat_pose = new_quat_pose.reshape(1, 24, 4)

        # TODO, why we needs normalization?
        new_quat_pose = new_quat_pose / torch.norm(new_quat_pose, p=2, dim=2, keepdim=True)
        new_quat_pose = new_quat_pose.reshape(1, 96)

        return new_quat_pose

    @staticmethod
    def sample_verts(verts, sample_num):
        """

        Args:
            verts (torch.Tensor): (N, 3);
            sample_num (int): number of sampled points;

        Returns:
            sampled_verts (np.ndarray): (sample_num, 3)

        """

        # sampled_idx = random.sample(range(verts.shape[0]), sample_num)
        # sampled_idx = torch.Tensor(sampled_idx).long().cuda()
        # sampled_verts = verts[sampled_idx]

        # np.random.sample is faster than random.sample, see https://blog.csdn.net/sunnyyan/article/details/83410233
        sampled_idx = np.random.choice(verts.shape[0], sample_num, replace=False)
        sampled_verts = verts[sampled_idx]

        return sampled_verts

    @staticmethod
    def compute_recons_loss(smpl_verts, smpl_faces, pifu_verts):
        """

        Args:
            smpl_verts (torch.tensor): (6890, 3);
            smpl_faces (torch.tensor): (13776, 3);
            pifu_verts (torch.tensor): (N, 3), here N is the batch size if sampled.

        Returns:
            recons_loss (torch.tensor): the scalar of the reconstructive loss.

        """

        # (13776, 3(faces), 3(xyz))
        smpl_faces_verts = smpl_verts[smpl_faces]
        # (13776, 3(xyz))
        smpl_faces_center = smpl_faces_verts.mean(dim=1)
        # (batch_size, 13776) dis of faces
        pifu_verts_dis = ((pifu_verts[:, None] - smpl_faces_center[None, :]) ** 2).mean(dim=2)

        recons_loss = pifu_verts_dis.min(dim=1)[0].mean()
        return recons_loss

    @staticmethod
    def compute_deform_reg_loss(deform_verts, smpl_verts_weights):
        """

        Args:
            deform_verts (torch.tensor): (6890, 3)
            smpl_verts_weights (torch.tensor): (6890,)

        Returns:
            loss (torch.tensor): scalar of the loss.

        """

        return ((deform_verts ** 2) * smpl_verts_weights[:, None]).mean()

    @staticmethod
    def compute_y_reg_loss(deform_verts):
        """

        Args:
            deform_verts (torch.tensor): (6890, 3)

        Returns:
            loss (torch.tensor): the scalar of the loss.
        """

        return (deform_verts[:, 1] ** 2).mean()

