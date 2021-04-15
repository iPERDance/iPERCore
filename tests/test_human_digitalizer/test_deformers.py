# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import unittest
import numpy as np
import torch

from tqdm import tqdm

from iPERCore.tools.human_digitalizer import deformers
from iPERCore.tools.human_digitalizer import renders
from iPERCore.tools.human_digitalizer.bodynets import SMPL, SMPLH
from iPERCore.tools.utils.filesio.persistence import load_pickle_file

from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer


visualizer = VisdomVisualizer(
    env='test_deformers',
    ip='http://10.10.10.100', port=31102
)


IMAGE_SIZE = 512
device = torch.device("cuda:0")
smpl = SMPL(model_path="assets/checkpoints/pose3d/smpl_model.pkl").to(device)
smplh = SMPLH(model_path="./assets/checkpoints/pose3d/smpl_model_with_hand_v2.pkl").to(device)

render = renders.SMPLRenderer(image_size=IMAGE_SIZE).to(device)
render.set_ambient_light()
texs = render.color_textures()[None].to(device)


def cloth_link_animate_visual(links_ids, cams, pose, shape, ref_smpl_path):
    """

    Args:
        links_ids:
        cams:
        pose:
        shape:
        ref_smpl_path (str):

    Returns:

    """
    global smpl, visualizer, render, device, texs

    cams = torch.tensor(cams).float().to(device)
    pose = torch.tensor(pose).float().to(device)
    shape = torch.tensor(shape).float().to(device)

    src_verts, _, _ = smpl.forward(shape, pose, offsets=0, links_ids=links_ids, get_skin=True)
    src_img, _ = render.render(cams, src_verts, texs)

    visualizer.vis_named_img("src_img", src_img)

    ref_smpl_info = load_pickle_file(ref_smpl_path)
    ref_cams = torch.tensor(ref_smpl_info["cams"]).float().to(device)
    ref_poses = torch.tensor(ref_smpl_info["pose"]).float().to(device)
    ref_shapes = torch.tensor(ref_smpl_info["shape"]).float().to(device)

    length = ref_poses.shape[0]

    for i in tqdm(range(length)):
        ref_pose = ref_poses[i:i+1]

        animate_verts, _, _ = smpl.forward(shape, ref_pose, offsets=0, links_ids=links_ids, get_skin=True)
        animate_img, _ = render.render(cams, animate_verts, texs)

        visualizer.vis_named_img("animate_img", animate_img)


class TestDeformers(unittest.TestCase):

    def test_01_clothlinks_deformer(self):
        device = torch.device("cuda:0")
        src_path = "/root/projects/iPERDance/iPERDanceCore-dev/tests/debug/primitives/skirts/processed/images/skirts.png"
        smpl_path = "/root/projects/iPERDance/iPERDanceCore-dev/tests/debug/primitives/skirts/processed/vid_info.pkl"

        smpls_data = load_pickle_file(smpl_path)["processed_pose3d"]

        ref_smpl_pkl = "/p300/projects/iPERDance/experiments/primitives/Av37667655_2.mp4/processed/pose_shape.pkl"

        cloth_link = deformers.ClothSmplLinkDeformer(
            cloth_parse_ckpt_path="./assets/checkpoints/mattors/exp-schp-lip.pth",
            smpl_model="assets/checkpoints/pose3d/smpl_model.pkl",
            part_path="assets/configs/pose3d/smpl_part_info.json",
            device=device
        )

        init_smpls = np.concatenate([smpls_data["cams"], smpls_data["pose"], smpls_data["shape"]], axis=1)

        has_linked, linked_ids = cloth_link.find_links(src_path, init_smpls)

        print(f"has_linked = {has_linked}")

        if has_linked:
            cloth_link_animate_visual(linked_ids,
                                      cams=smpls_data["cams"],
                                      pose=smpls_data["pose"],
                                      shape=smpls_data["shape"],
                                      ref_smpl_path=ref_smpl_pkl)


if __name__ == '__main__':
    unittest.main()
