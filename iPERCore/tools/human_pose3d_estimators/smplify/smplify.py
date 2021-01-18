# original file comes from Copyright (c) 2019, University of Pennsylvania, Max Planck Institute for Intelligent Systems. All rights reserved.
# original file: https://github.com/nkolot/SPIN/blob/master/smplify/smplify.py

import torch
from tqdm import tqdm

from iPERCore.tools.human_digitalizer.smplx import SMPL

from .losses import camera_fitting_loss, body_fitting_loss, temporal_body_fitting_loss

# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from .prior import MaxMixturePrior


class SMPLify(object):
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 model_path="./assets/pretrains/smpl_model.pkl",
                 prior_folder="./assets/pretrains",
                 step_size=1e-2,
                 batch_size=66,
                 num_iters=100,
                 focal_length=5000,
                 device=torch.device("cuda:0")):

        # Store options
        self.device = device
        self.focal_length = focal_length
        self.step_size = step_size

        # Ignore the the following joints for the fitting process
        # ign_joints = ["OP Neck", "OP RHip", "OP LHip", "Right Hip", "Left Hip"]
        # self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]
        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder=prior_folder,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # Load SMPL model
        self.smpl = SMPL(model_path=model_path,
                         batch_size=batch_size,
                         create_transl=False).to(self.device)

    def __call__(self, init_pose, init_betas, init_cam_t, camera_center, keypoints_2d,
                 opt_cam=False, use_lbfgs=False, use_temporal=False):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate;
            init_betas: SMPL betas estimate;
            init_cam_t: Camera translation estimate;
            camera_center: Camera center location;
            keypoints_2d: Keypoints used for the optimization;
            opt_cam (bool): Whether optimize camera or not;
            use_lbfgs (bool): Use lbfgs or not;
            use_temporal (bool): Use temporal optimization or not.
        Returns:
            results (dict): the result information, and it contains:
                --vertices: Vertices of optimized shape
                --joints: 3D joints of optimized shape
                --pose: SMPL pose parameters of optimized shape
                --betas: SMPL beta parameters of optimized shape
                --camera_translation: Camera translation
                --reprojection_loss: Final joint reprojection loss
        """

        if use_temporal:
            body_opt_func = temporal_body_fitting_loss
        else:
            body_opt_func = body_fitting_loss

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        if opt_cam:
            body_pose.requires_grad = False
            betas.requires_grad = False
            global_orient.requires_grad = True
            camera_translation.requires_grad = True

            camera_opt_params = [global_orient, camera_translation]

            if use_lbfgs:
                camera_optimizer = torch.optim.LBFGS(camera_opt_params, max_iter=20,
                                                     lr=self.step_size, line_search_fn="strong_wolfe")

                for i in range(self.num_iters):
                    def closure():
                        camera_optimizer.zero_grad()
                        smpl_output = self.smpl(global_orient=global_orient,
                                                body_pose=body_pose,
                                                betas=betas)
                        model_joints = smpl_output.joints
                        loss = camera_fitting_loss(model_joints, camera_translation,
                                                   init_cam_t, camera_center,
                                                   joints_2d, joints_conf, focal_length=self.focal_length)
                        loss.backward()
                        return loss

                    camera_optimizer.step(closure)

            else:
                camera_optimizer = torch.optim.Adam(
                    camera_opt_params, lr=self.step_size,
                    betas=(0.9, 0.999)
                )

                for i in tqdm(range(self.num_iters)):
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas)
                    model_joints = smpl_output.joints
                    loss = camera_fitting_loss(model_joints, camera_translation,
                                               init_cam_t, camera_center,
                                               joints_2d, joints_conf, focal_length=self.focal_length)
                    camera_optimizer.zero_grad()
                    loss.backward()
                    camera_optimizer.step()

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [body_pose, betas, global_orient]

        # For joints ignored during fitting, set the confidence to 0
        # joints_conf[:, self.ign_joints] = 0.

        if use_lbfgs:
            body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=20,
                                               lr=self.step_size, line_search_fn="strong_wolfe")
            for i in range(self.num_iters):
                def closure():
                    body_optimizer.zero_grad()
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas)
                    model_joints = smpl_output.joints

                    loss = body_opt_func(body_pose, betas, model_joints, camera_translation, camera_center,
                                         joints_2d, joints_conf, self.pose_prior,
                                         focal_length=self.focal_length)
                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:

            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in tqdm(range(self.num_iters)):
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints
                loss = body_opt_func(body_pose, betas, model_joints, camera_translation, camera_center,
                                     joints_2d, joints_conf, self.pose_prior,
                                     focal_length=self.focal_length)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            reprojection_loss = body_opt_func(body_pose, betas, model_joints, camera_translation, camera_center,
                                              joints_2d, joints_conf, self.pose_prior,
                                              focal_length=self.focal_length,
                                              output="reprojection")

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        results = {
            "vertices": vertices,
            "joints": joints,
            "pose": pose,
            "betas": betas,
            "camera_translation": camera_translation,
            "reprojection_loss": reprojection_loss
        }

        return results
