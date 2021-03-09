# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

"""

Assume that we put the iPER data into $iPER_root_dir

1. Download the iPER dataset, https://onedrive.live.com/?authkey=%21AJL%5FNAQMkdXGPlA&id=3705E349C336415F%2188052&cid=3705E349C336415F
    1.1 download the `images_HD.tar.gz`, and move them into $iPER_root_dir
    1.2 download the `train.txt`, and move them into $iPER_root_dir
    1.3 download the `val.txt`, and move them into $iPER_root_dir
    1.4 tar -xzvf `$iPER_root_dir/images_HD.tar.gz` into $iPER_root_dir

   The file structure of $iPER_root_dir will be:

   $iPER_root_dir:
        |-- images_HD
        |   |-- 001
        |   |   |-- 1
        |   |   |   |-- 1
        |   |   |   `-- 2
        |   |   |-- 10
        |   |   |   |-- 1
        |   |   |   `-- 2
        |   |-- 002
        |   |   `-- 1
        |   |       |-- 1
        |   |       `-- 2
        |   |-- 003
        |   |   `-- 1
        |   |       |-- 1
        |   |       `-- 2
        |   |-- 011
        |   |   `-- 1
        |   |       |-- 1
        |   |       `-- 2
        |-- images_HD.tar.gz
        |-- train.txt
        |-- val.txt

2. Preprocess all videos/frame sequences in $iPER_root_dir/images_HD.


3. # TODO, Reorganize the processed data for evaluations

"""

import os
import subprocess
import argparse
import glob
from tqdm import tqdm

from iPERCore.services.options.options_setup import setup
from iPERCore.services.options.process_info import ProcessInfo
from iPERCore.services.preprocess import human_estimate, digital_deform


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True, help="the root directory of iPER dataset.")
parser.add_argument("--gpu_ids", type=str, default="9", help="the gpu ids.")
parser.add_argument("--image_size", type=int, default=512, help="the image size.")
parser.add_argument("--model_id", type=str, default="iPER_Preprocess", help="the renamed model.")
parser.add_argument("--Preprocess.Cropper.src_crop_factor", type=float, default=0, help="directly resize on iPER.")
parser.add_argument("--ref_path", type=str, default="", help="set this empty when preprocessing training dataset.")

args = parser.parse_args()
args.cfg_path = os.path.join("./assets", "configs", "deploy.toml")

iPER_root_dir = args.output_dir
iPER_images_dir = os.path.join(iPER_root_dir, "images_HD")
iPER_smpls_dir = os.path.join(iPER_root_dir, "smpls")
iPER_train_txt = os.path.join(iPER_root_dir, "train.txt")
iPER_val_txt = os.path.join(iPER_root_dir, "val.txt")


def get_video_dirs(txt_file):
    vid_names = []
    with open(txt_file, "r") as reader:
        for line in reader:
            line = line.rstrip()
            vid_names.append(line)

    return vid_names


def prepare_src_path(video_names):
    global iPER_images_dir

    template_path = "path?={path},name?={name}"

    src_paths = []
    for vid_name in video_names:
        vid_img_dir = os.path.join(iPER_images_dir, vid_name)
        assert os.path.exists(vid_img_dir)

        path = template_path.format(path=vid_img_dir, name=vid_name)
        src_paths.append(path)

        print(path)

    return src_paths


def download():
    pass


def process_data():
    global args, iPER_train_txt, iPER_val_txt

    # 1. dumps videos to frames

    # 1. prepreocess
    vid_names = get_video_dirs(iPER_train_txt) + get_video_dirs(iPER_val_txt)
    src_paths = prepare_src_path(vid_names)

    args.src_path = "|".join(src_paths)

    # print(args.src_path)

    # set this as empty when preprocessing the training dataset.
    args.ref_path = ""

    cfg = setup(args)

    # 1. human estimation, including 2D pose, tracking, 3D pose, parsing, and front estimation.
    human_estimate(opt=cfg)

    # 2. digital deformation.
    digital_deform(opt=cfg)

    # 3. check
    meta_src_proc = cfg.meta_data["meta_src"]
    invalid_meta_process = []
    for meta_proc in meta_src_proc:
        process_info = ProcessInfo(meta_proc)
        process_info.deserialize()

        # check it has been processed successfully
        if not process_info.check_has_been_processed(process_info.vid_infos, verbose=False):
            invalid_meta_process.append(meta_proc)

    num_invalid = len(invalid_meta_process)
    if num_invalid > 0:
        for meta_proc in invalid_meta_process:
            print(f"invalid meta proc {meta_proc}")

    else:
        print(f"process successfully.")


def reorganize():
    # TODO
    pass


if __name__ == "__main__":
    download()
    process_data()
    # reorganize()
