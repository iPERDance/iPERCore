# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

"""

Assume that we put the iPER data into $iPER_root_dir

1. Download the iPER dataset, http://download.impersonator.org:20086/datasets/iPER
    1.1 download the all split files in http://download.impersonator.org:20086/datasets/iPER/images_HD,
        and merge all the split files into `images_HD.tar.gz`, and move it into $iPER_root_dir
    1.2 download the `train.txt`, and move them into $iPER_root_dir
    1.3 download the `val.txt`, and move them into $iPER_root_dir
    1.4 tar -xzvf `$iPER_root_dir/images_HD.tar.gz` into $iPER_root_dir

   The file structure of $iPER_root_dir will be:

   $iPER_root_dir:
        |-- images_HD.tar.gz
        |-- train.txt
        |-- val.txt

        |-- splits
        |   |-- images_HD.tar.gz.aa

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

2. Preprocess all videos/frame sequences in $iPER_root_dir/images_HD.


3. # TODO, Reorganize the processed data for evaluations

"""

import os
import subprocess
import argparse
from tqdm import tqdm

from iPERCore.services.options.options_setup import setup
from iPERCore.services.options.process_info import ProcessInfo
from iPERCore.services.preprocess import human_estimate, digital_deform
from iPERCore.tools.utils.filesio.persistence import mkdir

from scripts.train.download_utils import robust_download_from_url, download_from_url_to_file, raise_error


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True, help="the root directory of iPER dataset.")
parser.add_argument("--gpu_ids", type=str, default="0", help="the gpu ids.")
parser.add_argument("--image_size", type=int, default=512, help="the image size.")
parser.add_argument("--model_id", type=str, default="iPER_Preprocess", help="the renamed model.")
parser.add_argument("--Preprocess.Cropper.src_crop_factor", type=float, default=0, help="directly resize on iPER.")
parser.add_argument("--ref_path", type=str, default="", help="set this empty when preprocessing training dataset.")

args = parser.parse_args()
args.cfg_path = os.path.join("./assets", "configs", "deploy.toml")

iPER_root_dir = mkdir(args.output_dir)
iPER_images_dir = mkdir(os.path.join(iPER_root_dir, "images_HD"))
iPER_images_splits_dir = mkdir(os.path.join(iPER_root_dir, "splits"))
iPER_images_merged_tar_gz = os.path.join(iPER_root_dir, "images_HD.tar.gz")
iPER_smpls_dir = os.path.join(iPER_root_dir, "smpls")
iPER_train_txt = os.path.join(iPER_root_dir, "train.txt")
iPER_val_txt = os.path.join(iPER_root_dir, "val.txt")
iPER_images_HD_split_txt = os.path.join(iPER_root_dir, "images_HD_splits.txt")

DOWNLOAD_URL = "http://download.impersonator.org:20086/"
iPER_url = "http://download.impersonator.org:20086/datasets/iPER"
iPER_images_HD_url = os.path.join(iPER_url, "images_HD")
iPER_images_HD_split_txt_url = os.path.join(iPER_url, "images_HD_splits.txt")
iPER_train_txt_url = os.path.join(iPER_url, "train.txt")
iPER_val_txt_url = os.path.join(iPER_url, "val.txt")


def get_video_dirs(txt_file):
    vid_names = []
    with open(txt_file, "r") as reader:
        for line in reader:
            line = line.rstrip()
            vid_names.append(line)

    return vid_names


def check_has_downloaded():
    global iPER_images_dir, iPER_train_txt, iPER_val_txt

    has_download = os.path.exists(iPER_train_txt) and os.path.exists(iPER_val_txt)

    if has_download:
        train_vid_names = get_video_dirs(iPER_train_txt)
        val_vid_names = get_video_dirs(iPER_val_txt)

        all_vid_names = train_vid_names + val_vid_names

        for vid_name in all_vid_names:
            vid_path = os.path.join(iPER_images_dir, vid_name)
            print(vid_path)
            if not os.path.exists(vid_path) or len(os.listdir(vid_path)) == 0:
                has_download = False
                break

    return has_download


def download():
    global iPER_images_HD_url, iPER_images_HD_split_txt_url, iPER_train_txt_url, iPER_val_txt_url, \
        iPER_images_HD_split_txt, iPER_root_dir, iPER_images_dir, iPER_images_splits_dir, iPER_train_txt, iPER_val_txt

    print(f"Download all stuffs from {iPER_url}")

    # 1. download `iPER_images_HD_split_txt_url`
    success = download_from_url_to_file(iPER_images_HD_split_txt_url, iPER_images_HD_split_txt)

    if success and os.path.exists(iPER_images_HD_split_txt):
        download_success = True
    else:
        download_success = False
        raise_error(msg=f"Download {iPER_images_HD_split_txt_url} failed.")

    # 2. download all splits files from `iPER_images_HD_url`
    images_HD_splits_names = []
    with open(iPER_images_HD_split_txt, "r") as reader:
        for line in reader:
            line = line.rstrip()
            images_HD_splits_names.append(line)

    success_downloaded_splits_names = set()
    for split_name in tqdm(images_HD_splits_names):
        split_url = os.path.join(iPER_images_HD_url, split_name)
        target_split_file_path = os.path.join(iPER_images_splits_dir, split_name)

        success = robust_download_from_url(split_url, target_split_file_path, attempt_times=10)

        if success and os.path.exists(target_split_file_path):
            success_downloaded_splits_names.add(split_name)
            print(f"Download {split_name} successfully, and save it into {target_split_file_path}.")
        else:
            download_success = False
            if os.path.exists(target_split_file_path):
                os.remove(target_split_file_path)

            raise_error(f"Download {split_name} failed.")

    # 3. download train.txt and val.txt
    success = download_from_url_to_file(iPER_train_txt_url, iPER_train_txt)
    if not success or not os.path.join(iPER_train_txt):
        download_success = False
        raise_error(f"Download {iPER_train_txt_url} failed.")

    success = download_from_url_to_file(iPER_val_txt_url, iPER_val_txt)
    if not success or not os.path.join(iPER_val_txt):
        download_success = False
        raise_error(f"Download {iPER_val_txt_url} failed.")

    # 4. merge all splits files
    if not os.path.exists(iPER_images_merged_tar_gz):
        cmd = f"cat {iPER_images_splits_dir}/images_HD.tar.gz.*  >  {iPER_images_merged_tar_gz}"
        print(cmd)
        subprocess.run(cmd, shell=True)

    # 5. tar -xzvf images_HD.tar.gz
    if not check_has_downloaded():
        cmd = f"tar -xzvf {iPER_images_merged_tar_gz} -C {iPER_root_dir}"
        print(cmd)
        subprocess.run(cmd, shell=True)

    return download_success


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
    print(f"Check all videos have been processed...")
    for meta_proc in tqdm(meta_src_proc):
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
    # TODO, support evaluations
    pass


if __name__ == "__main__":
    is_download = check_has_downloaded()

    is_download_success = True
    if not is_download:
        is_download_success = download()

    if is_download_success:
        process_data()
        reorganize()
