# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch

from .sil_deformer import SilhouetteDeformer
from .clothlinks_deformer import ClothSmplLinkDeformer


def run_sil2smpl_offsets(obs_sils, init_smpls, image_size, device=torch.device("cuda:0"),
                         visualizer=None, visual_poses=None):
    """

    Args:
        obs_sils (np.ndarray):
        init_smpls (np.ndarray):
        image_size (int):
        device (torch.device):
        visualizer (None or Visualizer):
        visual_poses (None or np.ndarray):

    Returns:

    """
    # 1. define Deformer Solver
    deform_solver = SilhouetteDeformer(image_size=image_size, device=device)

    # 2. format inputs for SilhouetteDeformer.solve()
    cam = init_smpls[:, 0:3]
    pose = init_smpls[:, 3:-10]
    shape = init_smpls[:, -10:]

    obs = {
        "sil": obs_sils,
        "cam": cam,
        "pose": pose,
        "shape": shape
    }

    # 3. solve the offsets
    offsets = deform_solver.solve(obs, visualizer, visual_poses).cpu().detach().numpy()

    return offsets
