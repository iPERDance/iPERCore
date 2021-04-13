# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch

from .sil_deformer import SilhouetteDeformer
from .clothlinks_deformer import ClothSmplLinkDeformer
from .pifuhd2smpl_deformer import PifuHD2SMPLDeformer


def run_sil2smpl_offsets(obs_sils, init_smpls, image_size, device=torch.device("cuda:0"),
                         visualizer=None):
    """

    Args:
        obs_sils (np.ndarray):
        init_smpls (np.ndarray):
        image_size (int):
        device (torch.device):
        visualizer (None or Visualizer):

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
    offsets, new_smpl = deform_solver.solve(obs, visualizer)
    offsets = offsets.cpu().detach().numpy()
    new_smpl = new_smpl.cpu().detach().numpy()

    return offsets, new_smpl
