# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
from easydict import EasyDict
from pprint import pprint
import platform

from iPERCore.tools.utils.filesio.persistence import mkdir, load_toml_file, write_toml_file
from iPERCore.services.options.meta_info import parse_src_input, parse_ref_input, MetaProcess


def recursive_update_item(sub_models, val, cfg):
    """

    Args:
        sub_models (list of str):
        val (Any):
        cfg (EasyDict):

    Returns:
        cfg (EasyDict):
    """

    if len(sub_models) == 0:
        return cfg

    top_key = sub_models[0]
    if len(sub_models) == 1:
        if top_key not in cfg:
            cfg[top_key] = val
        else:
            # TODO, need a type-checking system
            old_val = cfg[top_key]
            if isinstance(old_val, int) and isinstance(val, str):
                cfg[top_key] = int(val)
            elif isinstance(old_val, float) and isinstance(val, str):
                cfg[top_key] = float(val)
            elif isinstance(old_val, bool) and isinstance(val, str):
                cfg[top_key] = bool(val)
            else:
                cfg[top_key] = val
    else:
        top_sub_cfg = cfg[top_key]
        cfg[top_key] = recursive_update_item(sub_models[1:], val, top_sub_cfg)

    return cfg


def update_cfg(opt, cfg):
    """

    Args:
        opt:
        cfg:

    Returns:

    """

    for key, val in opt.__dict__.items():
        # cfg[key] = val

        # "." as the separator
        sub_modules = key.split(".")
        recursive_update_item(sub_modules, val, cfg)


def update_extra_args(extra_args, cfg):
    """

    Args:
        extra_args:
        cfg:

    Returns:

    """
    if len(extra_args) < 2:
        return cfg
    elif len(extra_args) % 2 != 0:
        return cfg
    else:
        for i in range(0, len(extra_args), 2):
            # --num_source
            key = extra_args[i]
            if not key.startswith("--"):
                continue

            key = key[2:]
            val = extra_args[i + 1]

            sub_modules = key.split(".")
            recursive_update_item(sub_modules, val, cfg)

        return cfg


def load_cfg(cfg_path):
    """

    Parsing the configuration from self._opt.cfg_path, and add all configs to the self._opt.cfg.

    Returns:

    """
    cfg = EasyDict(load_toml_file(cfg_path))

    neural_render_cfg_path = cfg["neural_render_cfg_path"]
    neural_render_cfg = EasyDict(load_toml_file(neural_render_cfg_path))
    cfg.neural_render_cfg = neural_render_cfg

    return cfg


def load_inference_meta_data(cfg):
    """

    Set the meta directories:
        |--self._opt.output_dir:
            |--primitives:
                |--self._opt.primitives_id:
                    |--processed:
                    |--synthesis:
                        |--imitations:
                        |--swappers:
                        |--novel_views:
            |--models:
                |--self._opt.model_id:

    Returns:
        meta_data (EasyDict): the meta directories, it contains the followings:

    """

    meta_data = cfg.meta_data

    return meta_data


def load_meta_data(cfg):
    """

    Set the meta directories:
        |--self._opt.output_dir:
            |--models:
                |--self._opt.model_id:
                    |--opt_train.txt or opt_test.txt

    Returns:
        meta_data (EasyDict): the meta directories, it contains the followings:
            --checkpoints_dir:
    """

    output_dir = cfg.output_dir

    meta_data = EasyDict()
    meta_data["checkpoints_dir"] = mkdir(os.path.join(output_dir, "models", cfg.model_id))
    meta_data["personalized_ckpt_path"] = os.path.join(meta_data["checkpoints_dir"], "personalized.pth")
    meta_data["root_primitives_dir"] = mkdir(os.path.join(cfg.output_dir, "primitives"))
    meta_data["opt_path"] = os.path.join(
        meta_data["checkpoints_dir"], "opts.txt"
    )

    if "src_path" in cfg:
        src_inputs = parse_src_input(cfg.src_path)
        root_primitives_dir = meta_data["root_primitives_dir"]

        meta_src = []

        for meta_inp in src_inputs:
            meta_proc = MetaProcess(meta_inp, root_primitives_dir)
            meta_src.append(meta_proc)
        meta_data["meta_src"] = meta_src

    if "ref_path" in cfg:
        ref_inputs = parse_ref_input(cfg.ref_path)

        root_primitives_dir = meta_data["root_primitives_dir"]

        meta_ref = []

        for meta_inp in ref_inputs:
            meta_proc = MetaProcess(meta_inp, root_primitives_dir)
            meta_ref.append(meta_proc)
        meta_data["meta_ref"] = meta_ref

    return meta_data


def set_gpus(cfg):
    # set gpu, the gpu ids are 0 (single gpu) or 0,1,2,3 (multi-gpus)
    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    if len(cfg.gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # parse the "4,5,8,9" (multi-gpu if used) to ["4", "5", "8", "9"]
    cfg.gpu_ids = cfg.gpu_ids.split(",")


def set_multi_media(cfg):
    # set ffmpeg related flags
    os.environ["ffmpeg_vcodec"] = cfg.MultiMedia.ffmpeg.vcodec
    os.environ["ffmpeg_yuv420p"] = cfg.MultiMedia.ffmpeg.pix_fmt

    if platform.system() == "Windows":
        multimedia_ffmpeg_cfg = cfg.MultiMedia.ffmpeg["Windows"]
    else:
        multimedia_ffmpeg_cfg = cfg.MultiMedia.ffmpeg["Linux"]

    os.environ["ffmpeg_exe_path"] = multimedia_ffmpeg_cfg["ffmpeg_exe_path"]
    os.environ["ffprobe_exe_path"] = multimedia_ffmpeg_cfg["ffprobe_exe_path"]


def set_envs(cfg):

    # set gpus envs
    set_gpus(cfg)

    # set multi-media envs, including ffmpeg
    if "MultiMedia" in cfg:
        set_multi_media(cfg)


def save_cfg(cfg):
    if cfg.verbose:
        print("------------ Options -------------")
        pprint(cfg)
        print("-------------- End ----------------")

    file_name = cfg.meta_data.opt_path
    write_toml_file(file_name, cfg)


def setup(opt, extra_args=()):
    """

    Args:
        opt:
        extra_args (list or tuple):

    Returns:

    """
    # parse the configurations
    cfg = load_cfg(opt.cfg_path)
    update_extra_args(extra_args, cfg)
    update_cfg(opt, cfg)

    # get and set gpus
    set_envs(cfg)

    # set the meta data directories
    cfg.meta_data = load_meta_data(cfg)
    
    # print and save args to file
    save_cfg(cfg)

    return cfg
