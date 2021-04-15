# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
import numpy as np
import cv2

from iPERCore.tools.human_mattors.schp_parser import SchpMattor

from .link_utils import SmplLinker


class ClothSmplLinkDeformer(object):

    def __init__(self,
                 cloth_parse_ckpt_path="./assets/checkpoints/mattors/exp-schp-lip.pth",
                 smpl_model="assets/checkpoints/pose3d/smpl_model.pkl",
                 part_path="assets/configs/pose3d/smpl_part_info.json",
                 device=torch.device("cuda:0")):

        self.device = device
        self.smpl_link = SmplLinker(smpl_model=smpl_model, part_path=part_path, device=device)
        self.cloth_parser = SchpMattor(restore_weight=cloth_parse_ckpt_path, device=device)

    def find_links(self, img_path, init_smpls):
        """

        Args:
            img_path (str):
            init_smpls (np.ndarray or torch.tensor): (1, 85)

        Returns:
            flag (bool): if has found skirt or dress;
            from_verts_idx (list or np.ndarray):
            to_verts_idx (list or np.ndarray):
        """

        has_cloth, mask_outputs, _ = self.cloth_parser.run(None, None, src_file_list=[img_path],
                                                           target="skirt+dress", save_visual=False)

        if not has_cloth:
            return False, []

        mask = mask_outputs[0]

        if isinstance(mask, str):
            mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

        mask_h = mask.shape[0]
        skirt_y_int = np.argwhere(mask == 1)[-1][0]
        skirt_y = skirt_y_int / mask_h * 2 - 1

        if isinstance(init_smpls, np.ndarray):
            init_smpls = torch.from_numpy(init_smpls).float().to(self.device)
        elif init_smpls.deivce != self.device:
            init_smpls = init_smpls.to(self.device)

        cam = init_smpls[:, 0:3]
        pose = init_smpls[:, 3:-10]
        shape = init_smpls[:, -10:]

        from_verts_idx, to_verts_idx = self.smpl_link.link(cam, pose, shape, skirt_y, ret_tensor=False)

        linked_ids = np.stack([from_verts_idx, to_verts_idx], axis=1)

        return True, linked_ids
