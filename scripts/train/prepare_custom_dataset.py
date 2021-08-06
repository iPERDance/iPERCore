# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

"""

Assume that the root_dir has the following file structure:

$root_dir:
|----- aaaa:
|----------: image_01.png
|----------: image_02.png
...
|----------: image_0n.png
|----- bbbb:
...
|----- adfa:


Currently, it does not supports nests folders.

Example scripts:

python3 -m iPERCore.scripts.train.prepare_custom_dataset \
   --root_dir   /p300/tpami/custom_dataset/sources \
   --output_dir /p300/tpami/custom_dataset/outputs \
   --gpu_ids 0
"""

import os
import subprocess
import argparse
from tqdm import tqdm

from iPERCore.services.options.options_setup import setup
from iPERCore.services.options.process_info import ProcessInfo
from iPERCore.services.preprocess import human_estimate, digital_deform


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, required=True, help="the root directory dataset.")
parser.add_argument("--output_dir", type=str, required=True, help="the output directory dataset.")
parser.add_argument("--gpu_ids", type=str, default="0", help="the gpu ids.")
parser.add_argument("--image_size", type=int, default=512, help="the image size.")
parser.add_argument("--model_id", type=str, default="iPER_Preprocess", help="the renamed model.")
parser.add_argument("--ref_path", type=str, default="", help="set this empty when preprocessing training dataset.")

args = parser.parse_args()
args.cfg_path = os.path.join("./assets", "configs", "deploy.toml")

ROOT_DIR = args.root_dir


def prepare_src_path():
    global ROOT_DIR

    template_path = "path?={path},name?={name}"

    src_paths = []
    for vid_name in os.listdir(ROOT_DIR):
        vid_path = os.path.join(ROOT_DIR, vid_name)
        assert os.path.exists(vid_path)

        path = template_path.format(path=vid_path, name=vid_name)
        src_paths.append(path)

        print(path)

    return src_paths


def process_data():
    # 1. preprocess
    src_paths = prepare_src_path()

    args.src_path = "|".join(src_paths)

    print(args.src_path)

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
    # TODO, support evaluations
    pass


if __name__ == "__main__":
    process_data()
    reorganize()
