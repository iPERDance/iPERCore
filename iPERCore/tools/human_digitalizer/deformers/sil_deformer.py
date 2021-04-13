# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from iPERCore.tools.human_digitalizer.bodynets import SMPL
from iPERCore.tools.human_digitalizer.renders import SMPLRenderer
from iPERCore.tools.utils.morphology import morph

from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

from .utils import create_360_degree_T_Pose_view_smpl


class SilhouetteDeformer(object):
    def __init__(self, image_size=512, device=torch.device("cpu")):
        render = SMPLRenderer(
            image_size=image_size,
            face_path='./assets/checkpoints/pose3d/smpl_faces.npy',
            uv_map_path='./assets/checkpoints/pose3d/mapper_fim_enc.txt',
            fill_back=False
        ).to(device)

        smpl = SMPL(model_path="./assets/checkpoints/pose3d/smpl_model.pkl").to(device)

        self.render = render
        self.smpl = smpl
        self.device = device

        self.visual_render = SMPLRenderer(
            image_size=image_size,
            face_path='./assets/checkpoints/pose3d/smpl_faces.npy',
            uv_map_path='./assets/checkpoints/pose3d/mapper_fim_enc.txt',
            fill_back=False
        ).to(device)

        self.visual_render.set_ambient_light()

        visual_poses = create_360_degree_T_Pose_view_smpl(frame_num=5, ret_quat=False)
        self.visual_poses = torch.from_numpy(visual_poses.reshape(5, -1)).float().to(self.device)

        # (13776, 3)
        self.smpl_faces = render.smpl_faces
        self.nf = render.base_nf

    def solve(self, obs, visualizer=None):
        """
        Args:
            obs (dict): observations contains:
                --sil:
                --cam:
                --pose:
                --shape:
            visualizer:

        Returns:

        """

        print("{} use the parse observations to tune the offsets...".format(self.__class__.__name__))

        with torch.no_grad():
            obs_sil = torch.tensor(obs["sil"]).float().to(self.device)
            obs_cam = torch.tensor(obs["cam"]).float().to(self.device)
            obs_pose = torch.tensor(obs["pose"]).float().to(self.device)
            obs_shape = torch.tensor(obs["shape"]).float().to(self.device)
            # obs_sil = morph(obs_sil, ks=3, mode="dilate")
            obs_sil = morph(obs_sil, ks=3, mode="erode")
            obs_sil.squeeze_(dim=1)

            bs = obs_cam.shape[0]
            init_verts, _, _ = self.smpl(obs_shape, obs_pose, offsets=0, get_skin=True)
            faces = self.render.smpl_faces.repeat(bs, 1, 1)
            nv = init_verts.shape[1]

        offsets = nn.Parameter(torch.zeros((nv, 3)).to(self.device))
        pose = nn.Parameter(torch.tensor(obs["pose"]).float().to(self.device))
        shape = nn.Parameter(torch.tensor(obs["shape"]).float().to(self.device))

        total_steps = 500
        init_lr = 0.0002
        alpha_reg = 100
        w_edge = 100
        w_laplacian = 100
        w_normal = 10

        optimizer = torch.optim.Adam([offsets, pose, shape], lr=init_lr)
        crt_sil = nn.MSELoss()

        if visualizer is not None:
            textures = self.render.color_textures().repeat(bs, 1, 1, 1, 1, 1)
            textures = textures.to(self.device)
            num_visuals = self.visual_poses.shape[0]

        for i in tqdm(range(total_steps)):
            # verts, joints, Rs = self.smpl(obs_shape.detach(), obs_pose.detach(), offsets=offsets, get_skin=True)
            verts, joints, Rs = self.smpl(obs_shape, pose, offsets=offsets, get_skin=True)
            rd_sil = self.render.render_silhouettes(obs_cam.detach(), verts, faces=faces.detach())

            new_smpl_mesh = Meshes(verts=verts, faces=self.smpl_faces.expand(bs, self.nf, 3))
            edge_loss = mesh_edge_loss(new_smpl_mesh) * w_edge
            laplacian_loss = mesh_laplacian_smoothing(new_smpl_mesh, method="uniform") * w_laplacian
            normal_loss = mesh_normal_consistency(new_smpl_mesh) * w_normal

            loss = crt_sil(rd_sil, obs_sil) + alpha_reg * torch.mean(offsets ** 2) + edge_loss + laplacian_loss + normal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if visualizer is not None and i % 10 == 0:
                with torch.no_grad():
                    ids = np.random.choice(num_visuals, bs)
                    rand_pose = self.visual_poses[ids]
                    print(obs_shape.shape, rand_pose.shape)
                    t_verts, joints, Rs = self.smpl(obs_shape, rand_pose, offsets=offsets, get_skin=True)
                    rd, _ = self.visual_render.render(obs_cam, verts, textures, faces=faces, get_fim=False)
                    t_rd, _ = self.visual_render.render(obs_cam, t_verts, textures, faces=faces, get_fim=False)
                    visualizer.vis_named_img("rd_sil", rd_sil)
                    visualizer.vis_named_img("obs_sil", obs_sil)
                    visualizer.vis_named_img("t_render", t_rd)
                    visualizer.vis_named_img("render", rd)

                print("step = {}, loss = {:.6f}".format(i, loss.item()))

        new_smpl = torch.cat([obs_cam, pose, shape], dim=1)

        return offsets, new_smpl



# class SilhouetteDeformer(object):
#     def __init__(self, image_size=512, device=torch.device("cpu")):
#         render = SMPLRenderer(
#             image_size=image_size,
#             face_path='./assets/checkpoints/pose3d/smpl_faces.npy',
#             uv_map_path='./assets/checkpoints/pose3d/mapper_fim_enc.txt',
#             fill_back=False
#         ).to(device)
#
#         smpl = SMPL(model_path="./assets/checkpoints/pose3d/smpl_model.pkl").to(device)
#
#         self.render = render
#         self.smpl = smpl
#         self.device = device
#
#         self.visual_render = SMPLRenderer(
#             image_size=image_size,
#             face_path='./assets/checkpoints/pose3d/smpl_faces.npy',
#             uv_map_path='./assets/checkpoints/pose3d/mapper_fim_enc.txt',
#             fill_back=False
#         ).to(device)
#
#         self.visual_render.set_ambient_light()
#
#         visual_poses = create_360_degree_T_Pose_view_smpl(frame_num=5, ret_quat=False)
#         self.visual_poses = torch.from_numpy(visual_poses.reshape(5, -1)).float().to(self.device)
#
#     def solve(self, obs, visualizer=None):
#         """
#         Args:
#             obs (dict): observations contains:
#                 --sil:
#                 --cam:
#                 --pose:
#                 --shape:
#             visualizer:
#
#         Returns:
#
#         """
#
#         print("{} use the parse observations to tune the offsets...".format(self.__class__.__name__))
#
#         with torch.no_grad():
#             obs_sil = torch.tensor(obs["sil"]).float().to(self.device)
#             obs_cam = torch.tensor(obs["cam"]).float().to(self.device)
#             obs_pose = torch.tensor(obs["pose"]).float().to(self.device)
#             obs_shape = torch.tensor(obs["shape"]).float().to(self.device)
#             # obs_sil = morph(obs_sil, ks=3, mode="dilate")
#             obs_sil = morph(obs_sil, ks=3, mode="erode")
#             obs_sil.squeeze_(dim=1)
#
#             bs = obs_cam.shape[0]
#             init_verts, _, _ = self.smpl(obs_shape, obs_pose, offsets=0, get_skin=True)
#             faces = self.render.smpl_faces.repeat(bs, 1, 1)
#             nv = init_verts.shape[1]
#
#         offsets = nn.Parameter(torch.zeros((nv, 3)).to(self.device))
#         pose = nn.Parameter(torch.tensor(obs["pose"]).float().to(self.device))
#         shape = nn.Parameter(torch.tensor(obs["shape"]).float().to(self.device))
#
#         total_steps = 500
#         init_lr = 0.0002
#         alpha_reg = 1000
#
#         # total_steps = 500
#         # init_lr = 0.0001
#         # alpha_reg = 10000
#
#         optimizer = torch.optim.Adam([offsets, pose, shape], lr=init_lr)
#         crt_sil = nn.MSELoss()
#
#         if visualizer is not None:
#             textures = self.render.color_textures().repeat(bs, 1, 1, 1, 1, 1)
#             textures = textures.to(self.device)
#             num_visuals = self.visual_poses.shape[0]
#
#         for i in tqdm(range(total_steps)):
#             # verts, joints, Rs = self.smpl(obs_shape.detach(), obs_pose.detach(), offsets=offsets, get_skin=True)
#             verts, joints, Rs = self.smpl(shape, pose, offsets=offsets, get_skin=True)
#
#             rd_sil = self.render.render_silhouettes(obs_cam.detach(), verts, faces=faces.detach())
#             loss = crt_sil(rd_sil, obs_sil) + alpha_reg * torch.mean(offsets ** 2)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             if visualizer is not None and i % 10 == 0:
#                 with torch.no_grad():
#                     ids = np.random.choice(num_visuals, bs)
#                     rand_pose = self.visual_poses[ids]
#                     print(obs_shape.shape, rand_pose.shape)
#                     verts, joints, Rs = self.smpl(obs_shape, rand_pose, offsets=offsets, get_skin=True)
#                     rd, _ = self.visual_render.render(obs_cam, verts, textures, faces=faces, get_fim=False)
#                     visualizer.vis_named_img("rd_sil", rd_sil)
#                     visualizer.vis_named_img("obs_sil", obs_sil)
#                     visualizer.vis_named_img("render", rd)
#
#                 print("step = {}, loss = {:.6f}".format(i, loss.item()))
#
#         return offsets

