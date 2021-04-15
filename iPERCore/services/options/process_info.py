# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import numpy as np
import warnings

from iPERCore.tools.utils.filesio.persistence import mkdir, load_pickle_file, write_pickle_file, clear_dir


class ProcessInfo(object):

    """
    The Processed Information.
    """

    def __init__(self, meta_process):
        """

        Args:
            meta_process (MetaProcess):
        """

        processed_dir = meta_process.processed_dir

        self.vid_infos = {
            "input_info": meta_process.get_info(),

            "src_img_dir": os.path.join(processed_dir, "orig_images"),
            "src_num_imgs": 0,
            "out_img_dir": mkdir(os.path.join(processed_dir, "images")),
            "out_bg_dir": mkdir(os.path.join(processed_dir, "background")),
            "out_actual_bg_dir": mkdir(os.path.join(processed_dir, "actual_background")),
            "out_parse_dir": mkdir(os.path.join(processed_dir, "parse")),
            "out_visual_path": os.path.join(processed_dir, "visual.mp4"),

            "has_run_detector": False,
            "has_run_cropper": False,
            "has_run_3dpose": False,
            "has_find_front": False,
            "has_run_parser": False,
            "has_run_inpaintor": False,
            "has_run_deform": False,
            "has_finished": False,

            "orig_shape": (),

            "valid_img_info": {
                "names": [],
                "ids": [],
                "crop_ids": [],         # crop to all indexes
                "pose3d_ids": [],       # pose3d to crop
                "parse_ids": [],        # parse to pose3d
                "stage": ""
            },

            "processed_pose2d": {
                "boxes_XYXY": [],
                "keypoints": []
            },

            "processed_cropper": {
                "crop_shape": (),
                "active_boxes_XYXY": [],

                "crop_boxes_XYXY": [],
                "crop_keypoints": []
            },

            "processed_pose3d": {
                "cams": [],
                "pose": [],
                "shape": [],
                "init_pose": [],
                "init_shape": []
            },

            "processed_front_info": {
                "ft": {
                    "body_num": [],
                    "face_num": [],
                    "ids": []
                },
                "bk": {
                    "body_num": [],
                    "face_num": [],
                    "ids": []
                }
            },

            "processed_parse": {
                "mask_suffix": "mask.png",
                "alpha_suffix": "alpha.png"
            },

            "processed_background": {
                "inpainted_suffix": "_inpainted.png",
                "replaced_suffix": "_replaced.png",
                "replace": False,
            },

            "processed_deform": {
                "links": None,
                "offsets": None
            }
        }

    def __getitem__(self, item):
        return self.vid_infos[item]

    def __setitem__(self, key, value):
        self.vid_infos[key] = value

    def __contains__(self, item):
        return item in self.vid_infos

    def __str__(self):
        _str = "----------------------ProcessInfo----------------------\n"
        _str += "meta_input:\n"

        input_info = self.vid_infos["input_info"]

        for meta_input_key, meta_input_val in input_info["meta_input"].items():
            _str += f"\t{meta_input_key}: {meta_input_val}\n"

        _str += f"processed_dir: {input_info['processed_dir']}\n"
        _str += f"vid_info_path: {input_info['vid_info_path']}\n"
        _str += f"has_finished: {self.vid_infos['has_finished']}\n"

        _str += "-------------------------------------------------------"

        return _str

    def __repr__(self):
        return self.__str__()

    def serialize(self):
        vid_info_path = self.vid_infos["input_info"]["vid_info_path"]
        write_pickle_file(vid_info_path, self.vid_infos)

    def deserialize(self):
        vid_info_path = self.vid_infos["input_info"]["vid_info_path"]
        if os.path.exists(vid_info_path):
            input_info = self.vid_infos["input_info"]
            self.vid_infos = load_pickle_file(vid_info_path)
            self.vid_infos["input_info"] = input_info

    def declare(self):
        clear_dir(self.vid_infos["input_info"]["processed_dir"])

    @staticmethod
    def check_has_been_processed(context, verbose=False):
        has_finished = context["has_finished"]

        if verbose:
            print(f"\thas_run_detector: {context['has_run_detector']}")
            print(f"\thas_run_cropper: {context['has_run_cropper']}")
            print(f"\thas_run_3dpose: {context['has_run_3dpose']}")
            print(f"\thas_find_front: {context['has_find_front']}")
            print(f"\thas_run_parser: {context['has_run_parser']}")
            print(f"\thas_run_inpaintor: {context['has_run_inpaintor']}")
            print(f"\thas_run_deform: {context['has_run_deform']}")
            print(f"\thas_finished: {context['has_finished']}")

        return has_finished

    def convert_to_src_info(self, num_source):
        src_infos = read_src_infos(self.vid_infos, num_source)
        return src_infos

    def convert_to_ref_info(self):
        ref_infos = read_ref_infos(self.vid_infos)
        return ref_infos

    def num_sources(self):
        return len(self.vid_infos["valid_img_info"]["ids"])


def read_ref_infos(vid_infos):

    # 1.1 directory stores the cropped images.
    out_img_dir = vid_infos["out_img_dir"]

    # 1.2 valid_img_info
    valid_img_info = vid_infos["valid_img_info"]
    valid_img_names = valid_img_info["names"]

    # 2. 3D pose information
    processed_pose3d = vid_infos["processed_pose3d"]
    smpls = np.concatenate(
        [processed_pose3d["cams"],
         processed_pose3d["pose"],
         processed_pose3d["shape"]], axis=-1
    )

    assert len(smpls) == len(valid_img_names), f"the length of smpls = {len(smpls)} != the length of " \
                                               f"images = {len(valid_img_names)}."

    formated_vid_infos = {
        "input_info": vid_infos["input_info"],
        "smpls": smpls,
        "images": [os.path.join(out_img_dir, img_name) for img_name in valid_img_names]
    }

    return formated_vid_infos


def read_src_infos(vid_infos, num_source, num_verts=6890, ignore_bg=False):
    """

    Args:
        vid_infos (dict):
        num_source:
        num_verts:
        ignore_bg (bool)

    Returns:
        formated_vid_infos (dict): the formated video information, and it contains the followings,
            --

    """

    # 1.1 directory stores the cropped images, inpainted background images, and matting images.
    out_img_dir = vid_infos["out_img_dir"]
    out_bg_dir = vid_infos["out_bg_dir"]
    out_parse_dir = vid_infos["out_parse_dir"]

    # 1.2 valid_img_info
    valid_img_info = vid_infos["valid_img_info"]
    valid_img_names = valid_img_info["names"].copy()
    # valid_img_ids = valid_img_info["ids"].copy()

    # 2. 3D pose information
    processed_pose3d = vid_infos["processed_pose3d"]
    estimated_smpls = np.concatenate(
        [processed_pose3d["cams"],
         processed_pose3d["pose"],
         processed_pose3d["shape"]], axis=-1
    )

    # take the smpls with parsing results.
    smpls = estimated_smpls[valid_img_info["parse_ids"]]

    assert len(smpls) == len(valid_img_names), f"the length of smpls = {len(smpls)} != the length of " \
                                               f"images = {len(valid_img_names)}."

    # 3. front info
    front_info = vid_infos["processed_front_info"]

    if num_source == 1:
        src_ids = front_info["ft"]["ids"][0:1]
    else:
        half_ns = num_source // 2
        sample_ft_ids = front_info["ft"]["ids"][0:half_ns]
        sample_bk_ids = front_info["bk"]["ids"][0:half_ns]
        src_ids = sample_ft_ids + sample_bk_ids

    cur_src_num = len(src_ids)
    if cur_src_num < num_source:
        need_pad = num_source - cur_src_num
        pad_ids = np.random.choice(src_ids, need_pad)
        src_ids += list(pad_ids)

    # print(src_ids[0], valid_img_ids[0], valid_img_names[0])
    # if src_ids[0] != valid_img_ids[0]:
    #     warnings.warn(f"the first image {valid_img_names[0]} is not the frontal wise image.")
    #
    #     old = valid_img_ids[0]
    #     new = src_ids[0]
    #
    #     # swap `old` and `new`
    #     valid_img_ids[new], valid_img_ids[old] = valid_img_ids[old], valid_img_ids[new]
    #     valid_img_names[new], valid_img_names[old] = valid_img_names[old], valid_img_names[new]
    #     smpls[new], smpls[old] = smpls[old], smpls[new]

    # # 4. load parsing info list
    alpha_paths = []
    mask_paths = []
    for name in valid_img_names:
        name = name.split(".")[0]
        alpha_name = name + "_alpha.png"
        mask_name = name + "_mask.png"

        alpha_path = os.path.join(out_parse_dir, alpha_name)
        mask_path = os.path.join(out_parse_dir, mask_name)

        if not os.path.exists(alpha_path):
            warnings.warn(f"{alpha_path} does not exist.")
        else:
            alpha_paths.append(alpha_path)

        if os.path.exists(mask_path):
            mask_paths.append(mask_path)

    if not ignore_bg:
        # 6. actual background.
        out_actual_bg_dir = vid_infos["out_actual_bg_dir"]
        actual_bg_names = os.listdir(out_actual_bg_dir)
        num_bg = len(actual_bg_names)

        if num_bg > 1:
            warnings.warn(f"{out_actual_bg_dir} has more than 1 background images, "
                          f"and they are {out_actual_bg_dir}. Here we take the first background image.")

        if num_bg > 0:
            actual_bg_path = os.path.join(out_actual_bg_dir, actual_bg_names[0])
        else:
            actual_bg_path = None

        # 7. pre background inpainted images, if use additional inpaintor.
        processed_background = vid_infos["processed_background"]
        inpainted_suffix = processed_background["inpainted_suffix"]
        replaced_suffix = processed_background["replaced_suffix"]
        bg_replace = processed_background["replace"]

        inpainted_paths = []
        replaced_paths = []
        for ids in src_ids:
            name = valid_img_names[ids]
            name = name.split(".")[0]
            inpainted_name = name + inpainted_suffix
            replaced_name = name + replaced_suffix

            inpainted_path = os.path.join(out_bg_dir, inpainted_name)
            replaced_path = os.path.join(out_bg_dir, replaced_name)

            if not os.path.exists(inpainted_path):
                warnings.warn(f"{inpainted_path} does not exist.")
            else:
                inpainted_paths.append(inpainted_path)

            if bg_replace and not os.path.exists(replaced_path):
                warnings.warn(f"{replaced_path} does not exist.")
            else:
                replaced_paths.append(replaced_path)
    else:
        inpainted_paths = []
        replaced_paths = []
        actual_bg_path = []

    # 8. load digital deformation, including offsets and links.
    processed_deform = vid_infos["processed_deform"]

    links_ids = processed_deform["links"]
    offsets = processed_deform["offsets"]
    if offsets is None:
        offsets = np.zeros((num_verts, 3), dtype=np.float32)

    if links_ids is not None:
        num_links = links_ids.shape[0]
        links = np.zeros((num_verts, 3), dtype=np.int64)
        links[0:num_links, 0:2] = links_ids
        links[0:num_links, 2] = 1
    else:
        links = np.zeros((num_verts, 3), dtype=np.int64)

    formated_vid_infos = {
        "input_info": vid_infos["input_info"],
        "img_dir": out_img_dir,
        "bg_dir": out_bg_dir,
        "images": valid_img_names,
        "smpls": smpls,
        "offsets": offsets,
        "links": links,
        "length": len(smpls),
        "src_ids": src_ids,
        "ft_ids": front_info["ft"]["ids"],
        "bk_ids": front_info["bk"]["ids"],
        "alpha_paths": alpha_paths,
        "mask_paths": mask_paths,
        "inpainted_paths": inpainted_paths,
        "replaced_paths": replaced_paths,
        "actual_bg_path": actual_bg_path,
        "num_source": num_source,
    }

    return formated_vid_infos



