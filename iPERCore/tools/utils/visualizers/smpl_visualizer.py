# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from iPERCore.tools.human_digitalizer.bodynets import SMPL
from iPERCore.tools.human_digitalizer.renders import SMPLRenderer
from iPERCore.tools.utils.multimedia.video import convert_avi_to_mp4

from .skeleton_visualizer import draw_skeleton


def visual_pose3d_results(save_video_path, img_dir, smpls_info, parse_dir=None,
                          smpl_model="./assets/checkpoints/pose3d/smpl_model.pkl",
                          image_size=512, fps=25):
    """

    Args:
        save_video_path:
        img_dir:
        smpls_info:
        parse_dir:
        smpl_model:
        image_size:
        fps:

    Returns:

    """
    device = torch.device("cuda:0")
    render = SMPLRenderer(image_size=image_size).to(device)
    smpl = SMPL(smpl_model).to(device)

    render.set_ambient_light()
    texs = render.color_textures().to(device)[None]

    valid_img_names = smpls_info["valid_img_names"]
    all_init_cams = smpls_info["all_init_smpls"][:, 0:3]
    all_init_poses = smpls_info["all_init_smpls"][:, 3:-10]
    all_init_shapes = smpls_info["all_init_smpls"][:, -10:]
    all_opt_cams = smpls_info["all_opt_smpls"][:, 0:3]
    all_opt_poses = smpls_info["all_opt_smpls"][:, 3:-10]
    all_opt_shapes = smpls_info["all_opt_smpls"][:, -10:]
    all_keypoints = smpls_info["all_keypoints"]

    has_opt = len(all_opt_poses) > 0
    has_kps = all_keypoints is not None and len(all_keypoints) > 0

    def render_result(imgs, cams, poses, shapes):
        nonlocal texs

        verts, _, _ = smpl(beta=shapes, theta=poses, get_skin=True)
        rd_imgs, _ = render.render(cams, verts, texs)
        sil = render.render_silhouettes(cams, verts)[:, None].contiguous()
        masked_img = imgs * (1 - sil) + rd_imgs * sil
        return masked_img

    def visual_single_frame(i, image_name):
        nonlocal img_dir, parse_dir, all_opt_cams, all_opt_poses, all_opt_shapes, \
            all_init_cams, all_init_poses, all_init_shapes, has_opt

        im_path = os.path.join(img_dir, image_name)
        image = cv2.imread(im_path)

        if has_kps:
            joints = all_keypoints[i]
            image = draw_skeleton(image, joints, radius=6, transpose=False, threshold=0.25)

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float) / 255
        image = torch.tensor(image).float()[None].to(device)

        init_cams = torch.tensor(all_init_cams[i]).float()[None].to(device)
        init_pose = torch.tensor(all_init_poses[i]).float()[None].to(device)
        init_shape = torch.tensor(all_init_shapes[i]).float()[None].to(device)
        init_result = render_result(image, cams=init_cams, poses=init_pose, shapes=init_shape)

        fused_images = [image, init_result]

        if parse_dir is not None:
            alpha_path = os.path.join(parse_dir, image_name.split(".")[0] + "_alpha.png")

            if os.path.exists(alpha_path):
                alpha = cv2.imread(alpha_path)
                alpha = alpha.astype(np.float32) / 255
                alpha = np.transpose(alpha, (2, 0, 1))
                alpha = torch.from_numpy(alpha).to(device)
                alpha.unsqueeze_(0)
                fused_images.append(alpha)

        if has_opt:
            opt_cams = torch.tensor(all_opt_cams[i]).float()[None].to(device)
            opt_pose = torch.tensor(all_opt_poses[i]).float()[None].to(device)
            opt_shape = torch.tensor(all_opt_shapes[i]).float()[None].to(device)
            opt_result = render_result(image, cams=opt_cams, poses=opt_pose, shapes=opt_shape)
            fused_images.append(opt_result)

        num = len(fused_images)
        if num % 2 == 0:
            nrow = 2
        else:
            nrow = 3

        fused_images = torch.cat(fused_images, dim=0)
        fused_images = make_grid(fused_images, nrow=nrow, normalize=False)

        return fused_images

    if len(all_init_shapes) == 0:
        return

    first_image = visual_single_frame(0, valid_img_names[0]).cpu().numpy()
    height, width = first_image.shape[1:]

    tmp_avi_video_path = f"{save_video_path}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(tmp_avi_video_path, fourcc, fps, (width, height))

    for i, image_name in enumerate(tqdm(valid_img_names)):
        fused_image = visual_single_frame(i, image_name)
        fused_image = fused_image.cpu().numpy()
        fused_image = np.transpose(fused_image, (1, 2, 0))
        fused_image = fused_image * 255
        fused_image = fused_image.astype(np.uint8)

        videoWriter.write(fused_image)

    videoWriter.release()

    convert_avi_to_mp4(tmp_avi_video_path, save_video_path)
