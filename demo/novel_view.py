# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import os.path as osp
import platform
import argparse
import subprocess
import sys
import time

from iPERCore.services.options.options_setup import setup
from iPERCore.services.run_viewer import run_viewer

###############################################################################################
##                   Setting
###############################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_ids", type=str, default="0", help="the gpu ids.")
parser.add_argument("--image_size", type=int, default=512, help="the image size.")
parser.add_argument("--num_source", type=int, default=2, help="the number of sources.")
parser.add_argument("--output_dir", type=str, default="./results", help="the output directory.")
parser.add_argument("--assets_dir", type=str, default="./assets",
                    help="the assets directory. This is very important, and there are the configurations and "
                         "the all pre-trained checkpoints")

src_path_format = """the source input information. All source paths and it supports multiple paths,
    uses "|" as the separator between all paths.
    The format is "src_path_1|src_path_2|src_path_3".
    Each src_input is "path?=path1,name?=name1,bg_path?=bg_path1".
    It must contain "path". If "name" and "bg_path" are empty, they will be ignored.

    The "path" could be an image path, a path of a directory contains source images, and a video path.

    The "name" is the rename of this source input, if it is empty, we will ignore it,
    and use the filename of the path.

    The "bg_path" is the actual background path if provided, otherwise we will ignore it.

    There are several examples of formated source paths,

    1. "path?=path1,name?=name1,bg_path?=bg_path1|path?=path2,name?=name2,bg_path?=bg_path2",
    this input will be parsed as [{path: path1, name: name1, bg_path:bg_path1},
    {path: path2, name: name2, bg_path: bg_path2}];

    2. "path?=path1,name?=name1|path?=path2,name?=name2", this input will be parsed as
    [{path: path1, name:name1}, {path: path2, name: name2}];

    3. "path?=path1", this input will be parsed as [{path: path1}].

    4. "path1", this will be parsed as [{path: path1}].
"""

parser.add_argument("--model_id", type=str, default=f"model_{str(time.time())}", help="the renamed model.")
parser.add_argument("--src_path", type=str, required=True, help=src_path_format)
parser.add_argument("--T_pose", action="store_true", default=False,
                    help="view as T pose or not in human novel view synthesis.")

# args = parser.parse_args()
args, extra_args = parser.parse_known_args()

# symlink from the actual assets directory to this current directory
work_asserts_dir = os.path.join("./assets")
if not os.path.exists(work_asserts_dir):
    os.symlink(osp.abspath(args.assets_dir), osp.abspath(work_asserts_dir),
               target_is_directory=(platform.system() == "Windows"))

args.cfg_path = osp.join(work_asserts_dir, "configs", "deploy.toml")

if __name__ == "__main__":
    # run novel view
    # args.ref_path = ""
    # cfg = setup(args, extra_args)
    # run_viewer(cfg)

    # or use the system call wrapper
    cmd = [
        sys.executable, "-m", "iPERCore.services.run_viewer",
        "--cfg_path", args.cfg_path,
        "--gpu_ids", args.gpu_ids,
        "--image_size", str(args.image_size),
        "--num_source", str(args.num_source),
        "--output_dir", args.output_dir,
        "--model_id", args.model_id,
        "--src_path", args.src_path,
        "--ref_path", "",
    ]

    if args.T_pose:
        cmd += ["--T_pose"]

    cmd += extra_args
    subprocess.call(cmd)
