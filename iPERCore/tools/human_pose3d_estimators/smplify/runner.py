# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
from easydict import EasyDict

from .smplify import SMPLify
from .. import BasePose3dRefiner

from iPERCore.tools.utils.filesio.persistence import load_toml_file
from iPERCore.tools.utils.geometry.keypoints import KEYPOINTS_FORMATER


class SMPLifyRunner(BasePose3dRefiner):

    def __init__(self,
                 cfg_or_path, use_lbfgs=False,
                 joint_type="OpenPose-Body-25", device=torch.device("cpu")):

        """

        Args:
            cfg_or_path: the configuration of SMPLify, the default file is `./assets/configs/pose3d/smplify.toml`. It
                contains the followings parameters:

                    model_path = "./assets/pretrains/smpl_model.pkl"
                    prior_folder = "./assets/pretrains"

                    focal_length = 5000

                    num_smpl_joints=45
                    ignore_joints=["Neck", "RHip", "LHip"]

                    [LBFGS]
                    # the hyper-parameters of LBFGS optimizer
                    num_iters = 1
                    step_size = 1.0

                    [ADAM]
                    # the hyper-parameters of ADAM optimizer
                    num_iters = 100
                    step_size = 0.01

            use_lbfgs (bool): whether to use LBFGS or not;
            joint_type (str): the 2d joint type;
            device (torch.device):
        """

        self.device = device
        self.use_lbfgs = use_lbfgs

        if isinstance(cfg_or_path, str):
            cfg = EasyDict(load_toml_file(cfg_or_path))
        else:
            cfg = cfg_or_path

        if use_lbfgs:
            num_iters = cfg.LBFGS.num_iters
            step_size = cfg.LBFGS.step_size
        else:
            num_iters = cfg.ADAM.num_iters
            step_size = cfg.ADAM.step_size

        self.smplify = SMPLify(
            model_path=cfg.model_path,
            prior_folder=cfg.prior_folder,
            focal_length=cfg.focal_length,
            step_size=step_size, num_iters=num_iters,
            batch_size=1, device=device
        )

        self.formater = KEYPOINTS_FORMATER[joint_type](
            num_smpl_joints=cfg.num_smpl_joints,
            ignore_joints=cfg.ignore_joints
        )

    @property
    def focal_length(self):
        return self.smplify.focal_length

    def __call__(self, keypoints, pred_camera, pred_betas, pred_pose, proc_kps=True, temporal=True):
        """

        Args:
            keypoints (dict): the outputs of the OpenPose, Halpe, or COCO, it contains:
                --pose_keypoints_2d (np.ndarray): the scale is in the range of [-1, 1]
            pred_camera: torch.Tensor, (bs, 3), the camera output of the SPIN or HMR;
            pred_betas: torch.Tensor, (bs, 10), the shape output of the SPIN or HMR;
            pred_pose: torch.Tensor, (bs, 72), the pose output of the SPIN or HMR;
            proc_kps (bool): if it is true, then, we need to convert the keypoints from [-1, 1] to [0, 224]
            temporal (bool): if it is true, then, optimize with temporal smooth loss.

        Returns:
            opt_results (dict): the optimization information, and it contains,
                --new_opt_betas:
                --new_opt_pose:
                --new_opt_vertices:
        """

        if proc_kps:
            keypoints = self.formater.format_keypoints(keypoints, im_shape=None)

        pred_cam_t = torch.stack(
            [pred_camera[:, 1],
             pred_camera[:, 2],
             2 * self.focal_length / (224 * pred_camera[:, 0] + 1e-9)], dim=-1
        )

        bs = keypoints.shape[0]
        pred_pose[torch.isnan(pred_pose)] = 0.0

        smplify_results = self.smplify(
            pred_pose.detach(), pred_betas.detach(),
            pred_cam_t.detach(),
            0.5 * 224 * torch.ones(bs, 2, device=self.device),
            keypoints, opt_cam=False, use_lbfgs=self.use_lbfgs, use_temporal=temporal)

        opt_results = {
            "new_opt_pose": smplify_results["pose"],
            "new_opt_betas": smplify_results["betas"],
            "new_opt_vertices": smplify_results["vertices"]
        }

        return opt_results

    def run(self, *args, **kwargs):
        pass
