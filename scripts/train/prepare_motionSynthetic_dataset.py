# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

"""

Assume that we put the iPER data into $FashionVideo_root_dir

1. Download the FashionVideo dataset, https://vision.cs.ubc.ca/datasets/fashion/
    1.1 download the `fashion_train.txt` into $FashionVideo_root_dir:
        https://vision.cs.ubc.ca/datasets/fashion/resources/fashion_dataset/fashion_train.txt

    1.2 download the `fashion_test.txt` into $FashionVideo_root_dir:
        https://vision.cs.ubc.ca/datasets/fashion/resources/fashion_dataset/fashion_test.txt

    1.3 crawl each video in `fashion_train.txt`, as well as `fashion_test.txt` and
        save them into $FashionVideo_root_dir/videos

   The file structure of $FashionVideo_root_dir will be:

   $FashionVideo_root_dir:
        --fashion_train.txt
        --fashion_test.txt
        --videos:


2. Preprocess all videos in $FashionVideo_root_dir/videos.


3. Reorganize the processed data for evaluations, https://github.com/iPERDance/his_evaluators


"""

import os
import argparse
from tqdm import tqdm
import requests
import warnings
import math
import numpy as np
import torch
from typing import List, Tuple
import cv2
from concurrent.futures import ProcessPoolExecutor

from iPERCore.services.options.meta_info import MetaInputInfo, MetaProcess
from iPERCore.services.options.process_info import ProcessInfo
from iPERCore.tools.utils.filesio.persistence import mkdir, load_pickle_file, write_pickle_file
from iPERCore.tools.utils.filesio.cv_utils import read_cv2_img, save_cv2_img
from iPERCore.tools.utils.multimedia.video import video2frames
from iPERCore.tools.human_digitalizer.renders import SMPLRenderer
from iPERCore.tools.human_digitalizer.bodynets import SMPL


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True, help="the root directory of iPER dataset.")
parser.add_argument("--gpu_ids", type=str, default="9", help="the gpu ids.")

args = parser.parse_args()

# the render image size
IMAGE_SIZE = 1080
MotionSynthetic_root_dir = args.output_dir
MotionSynthetic_video_dir = mkdir(os.path.join(MotionSynthetic_root_dir, "videos"))
MotionSynthetic_poses_dir = mkdir(os.path.join(MotionSynthetic_root_dir, "poses"))
MotionSynthetic_processed_dir = mkdir(os.path.join(MotionSynthetic_root_dir, "primitives"))


def download():
    pass


def render_mask(smpl, render, cams, poses, shapes, offsets,
                img_dir, image_names, parse_out_dir, visual_path):
    """

    Args:
        smpl (SMPL):
        render (SMPLRenderer):
        cams:
        poses:
        shapes:
        offsets:
        img_dir:
        image_names:
        parse_out_dir:
        visual_path:

    Returns:

    """

    global IMAGE_SIZE

    length = cams.shape[0]

    device = torch.device("cuda:0")

    offsets = torch.tensor(offsets).float().to(device)
    textures = render.color_textures().to(device)[None]

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    videoWriter = cv2.VideoWriter(visual_path, fourcc, 25, (IMAGE_SIZE, IMAGE_SIZE))

    print(f"Preprocessing {img_dir}")
    for i in tqdm(range(length)):
        img_name = image_names[i]
        name = img_name.split(".")[0]

        image = read_cv2_img(os.path.join(img_dir, img_name))

        cam = torch.from_numpy(cams[i: i+1]).to(device)
        shape = torch.from_numpy(shapes[i: i+1]).to(device)
        pose = torch.from_numpy(poses[i: i+1]).to(device)

        verts, _, _ = smpl(shape, pose, offsets=offsets, get_skin=True)

        rd_imgs, _ = render.render(cam, verts, textures)
        mask = render.render_silhouettes(cam, verts)[0]
        mask.unsqueeze_(-1)

        # (h, w, 1)
        mask = mask.cpu().numpy()

        # (3, h, w)
        rd_imgs = rd_imgs.cpu().numpy()[0]

        # (h, w, 3)
        rd_imgs = np.transpose(rd_imgs, (1, 2, 0))
        rd_imgs = (rd_imgs + 1) / 2 * 255
        rd_imgs = rd_imgs.astype(np.uint8)
        overly_img = image * (1 - mask) + rd_imgs * mask

        parse_path = os.path.join(parse_out_dir, name + "_alpha.png")

        save_cv2_img(mask[:, :, 0] * 255, parse_path, transpose=False)

        overly_img = overly_img.astype(np.uint8)
        videoWriter.write(overly_img)

    videoWriter.release()


def prepare_process_info(vid_name):
    global MotionSynthetic_video_dir, MotionSynthetic_processed_dir

    vid_path = os.path.join(MotionSynthetic_video_dir, vid_name)

    name = vid_name.split(".")[0]
    meta_input = MetaInputInfo(
        path=vid_path,
        name=name
    )

    meta_process = MetaProcess(
        meta_input,
        root_primitives_dir=MotionSynthetic_processed_dir
    )

    process_info = ProcessInfo(meta_process)

    return process_info


def partition(gpu_ids):
    global MotionSynthetic_video_dir

    video_names = os.listdir(MotionSynthetic_video_dir)
    num_videos = len(video_names)
    num_gpus = len(gpu_ids)

    gpu_per_subs = int(math.ceil(num_videos / num_gpus))

    gpu_i = 0

    used_gpus = []
    all_process_partition = []
    for sub_i in range(0, num_videos, gpu_per_subs):
        sub_names_i = video_names[sub_i: sub_i + gpu_per_subs]

        single_gpu_process_partition = []
        for vid_name in sub_names_i:
            process_info = prepare_process_info(vid_name)
            single_gpu_process_partition.append(process_info)

        all_process_partition.append(single_gpu_process_partition)
        used_gpus.append(gpu_ids[gpu_i])

        gpu_i += 1

    return used_gpus, all_process_partition


def per_instance_func(smpl, render, process_info):
    """

    Args:
        smpl (SMPL):
        render (SMPLRenderer):
        process_info (ProcessInfo):

    Returns:

    """

    global MotionSynthetic_video_dir, MotionSynthetic_poses_dir

    # 1. dump video into frames and write indexes information into process_info["valid_img_info"]
    input_info = process_info["input_info"]
    name = input_info["meta_input"]["name"]

    vid_path = input_info["meta_input"]["path"]
    out_img_dir = process_info["out_img_dir"]
    video2frames(vid_path=vid_path, out_dir=out_img_dir)

    img_names = os.listdir(out_img_dir)
    img_names.sort()

    img_ids = list(range(len(img_names)))

    process_info["valid_img_info"]["names"] = img_names
    process_info["valid_img_info"]["ids"] = img_ids
    process_info["valid_img_info"]["crop_ids"] = img_ids
    process_info["valid_img_info"]["pose3d_ids"] = img_ids
    process_info["valid_img_info"]["parse_ids"] = img_ids
    process_info["valid_img_info"]["stage"] = "parser"

    # 2. write the smpl information into process_info["processed_pose3d"]
    smpl_path = os.path.join(MotionSynthetic_poses_dir, name, "pose_shape.pkl")

    """
    The smpl_info contains:
        --cams (np.ndarray): (length, 3);
        --pose (np.ndarray): (length, 72);
        --shape (np.ndarray): (1, 10);
        --ft_ids (list): (4,);
        --bk_ids (list): (4,);
        --views (list): (8,) = [0, 45, 315, 90, 180, 135, 225, 270];
        --offsets (np.ndarray): (6890, 3)
    """
    smpl_info = load_pickle_file(smpl_path)
    length = len(img_names)
    offsets = smpl_info["offsets"]

    assert length == smpl_info["cams"].shape[0], f"{length} != {len(smpl_info['cams'])}"

    # repeat (1, 10) to (length, 10)
    shape = np.tile(smpl_info["shape"], (length, 1))
    process_info["processed_pose3d"]["cams"] = smpl_info["cams"]
    process_info["processed_pose3d"]["pose"] = smpl_info["pose"]
    process_info["processed_pose3d"]["shape"] = shape

    # 3. write the frontal information into process_info["processed_front_info"]
    process_info["processed_front_info"]["ft"]["ids"] = smpl_info["ft_ids"] + smpl_info["bk_ids"]
    process_info["processed_front_info"]["bk"]["ids"] = smpl_info["bk_ids"] + smpl_info["ft_ids"]

    # 4. render mask
    parse_out_dir = process_info["out_parse_dir"]
    visual_path = process_info["out_visual_path"]

    # render_mask(
    #     smpl, render, smpl_info["cams"], smpl_info["pose"], shape, offsets,
    #     out_img_dir, img_names, parse_out_dir, visual_path
    # )

    process_info["has_run_detector"] = True
    process_info["has_run_cropper"] = True
    process_info["has_run_3dpose"] = True
    process_info["has_find_front"] = True
    process_info["has_run_parser"] = True
    process_info["has_run_inpaintor"] = True
    process_info["has_run_deform"] = True
    process_info["has_finished"] = True

    process_info.serialize()


def process_func(gpu_id, process_info_list):

    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = torch.device("cuda:0")
    render = SMPLRenderer(image_size=IMAGE_SIZE).to(device)
    smpl = SMPL().to(device)

    print(f"----------------------------gpu_id = {gpu_id}----------------------")
    for process_info in process_info_list:
        print(gpu_id, process_info)
        per_instance_func(smpl, render, process_info)


def process_data():
    global args

    gpu_ids = args.gpu_ids
    gpu_ids = gpu_ids.split(",")

    used_gpus, all_process_partition = partition(gpu_ids=gpu_ids)

    with ProcessPoolExecutor(max_workers=min(len(gpu_ids), os.cpu_count())) as pool:
        pool.map(process_func, used_gpus, all_process_partition)

    # for gpu_id, process_info_list in zip(used_gpus, all_process_partition):
    #     process_func(gpu_id, process_info_list)


def reorganize():
    # TODO
    pass


# def copy_offsets_smpl_info():
#     global MotionSynthetic_root_dir, MotionSynthetic_poses_dir
#
#     people_snapshot_dir = "/p300/tpami/datasets/round-1-datasets/motionSynthetic/people_snapshot_public"
#
#     vid_names = os.listdir(MotionSynthetic_poses_dir)
#
#     for vid_name in vid_names:
#         smpl_path = os.path.join(MotionSynthetic_poses_dir, vid_name, "pose_shape.pkl")
#         smpl_info = load_pickle_file(smpl_path)
#
#         if vid_name.startswith("PeopleSnapshot"):
#             people_snapshot_name = vid_name.split("_")[1]
#             people_snapshot_vid_dir = os.path.join(people_snapshot_dir, people_snapshot_name)
#
#             v_info = load_pickle_file(os.path.join(people_snapshot_vid_dir, "consensus.pkl"))
#             offsets = v_info["v_personal"]
#
#             smpl_info["offsets"] = offsets
#         else:
#             smpl_info["offsets"] = np.zeros((6890, 3), dtype=np.float32)
#
#         write_pickle_file(smpl_path, smpl_info)
#
#         print(smpl_path)


if __name__ == "__main__":
    # download()
    process_data()
    # copy_offsets_smpl_info()
