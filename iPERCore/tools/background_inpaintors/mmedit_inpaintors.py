# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import time
import cv2
import torch
import numpy as np
import os

from mmedit.apis import init_model, inpainting_inference, restoration_inference
from mmedit.core import tensor2img

from iPERCore.tools.utils.filesio.persistence import load_toml_file
from iPERCore.tools.utils.filesio.cv_utils import compute_scaled_size


class SuperResolutionInpaintors(object):

    def __init__(self, cfg_or_path,
                 device=torch.device("cuda:0")):

        """

        Args:
            cfg_or_path (str or dict):
            device (torch.device):
        """

        if isinstance(cfg_or_path, str):
            cfg = load_toml_file(cfg_or_path)
        else:
            cfg = cfg_or_path

        self.inpainting_control_size = cfg["inpainting_control_size"]

        """ deepfill_v2  """
        self.inpainting_cfg_path = cfg["inpainting_cfg_path"]
        self.inpainting_ckpt_path = cfg["inpainting_ckpt_path"]

        """ super-resolution"""
        self.sr_cfg_path = cfg["sr_cfg_path"]
        self.sr_ckpt_path = cfg["sr_ckpt_path"]
        self.temp_dir = cfg["temp_dir"]

        self.inpainting_model = init_model(self.inpainting_cfg_path,
                                           self.inpainting_ckpt_path, device=device.__str__())
        self.sr_model = init_model(self.sr_cfg_path, self.sr_ckpt_path, device=device.__str__())
        self.device = device
        self.cfg = cfg

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def run_sr(self):
        pass

    def run_inpainting(self, img_or_path, mask_or_path,
                       dilate_kernel_size=19, dilate_iter_num=3):
        """

        Args:
            img_or_path (str or np.ndarray): (h, w, 3) is in the range of [0, 255] with BGR channel;
            mask_or_path (str or np.ndarray): (h, w) is in the range of [0, 255], np.uint8;
            dilate_kernel_size (int): the kernel size of dilation;
            dilate_iter_num (int): the iterations of dilation;

        Returns:
            inpainting_result (np.ndarray): (h, w, 3), is in the range of [0, 255] with BGR channel.
        """

        # TODO, do not write the middle outputs to disk, and make them in memory.
        #  scaled_src_path, scaled_mask_path, scaled_inpainting_result_path

        img_name = str(time.time())
        img_path = os.path.join(self.temp_dir, img_name)

        if isinstance(img_or_path, str):
            src_img = cv2.imread(img_or_path)
        else:
            src_img = img_or_path.copy()

        """
        scaled image 
        """
        scaled_src_path = f"{img_path}_scaled.png"
        scaled_mask_path = f"{img_path}_mask.png"
        scaled_inpainting_result_path = f"{img_path}_inpainting.png"

        origin_h, origin_w = src_img.shape[:2]
        scaled_size = compute_scaled_size((origin_w, origin_h), control_size=self.inpainting_control_size)

        scaled_src_img = cv2.resize(src_img, scaled_size)
        cv2.imwrite(scaled_src_path, scaled_src_img)

        """
        dilate mask
        """
        if isinstance(mask_or_path, str):
            mask = cv2.imread(mask_or_path, cv2.IMREAD_GRAYSCALE)
        else:
            # mask = (mask * 255).astype(np.uint8)
            mask = mask_or_path

        scaled_mask = cv2.resize(mask, scaled_size, interpolation=cv2.INTER_NEAREST)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
        dilated_scaled_mask = cv2.dilate(scaled_mask, kernel, iterations=dilate_iter_num)
        cv2.imwrite(scaled_mask_path, dilated_scaled_mask)

        """
        inpainting result
        """
        scaled_result = inpainting_inference(self.inpainting_model, scaled_src_path, scaled_mask_path)
        # (h, w, 3) 0-255 bgr
        scaled_result = tensor2img(scaled_result, min_max=(-1, 1))[..., ::-1]
        cv2.imwrite(scaled_inpainting_result_path, scaled_result)

        """
        super-resolution
        """
        if self.cfg["use_sr"]:
            result = restoration_inference(self.sr_model, scaled_inpainting_result_path)
            result = tensor2img(result)
            result = cv2.resize(result, (origin_w, origin_h))
            result = result.astype(np.uint8)

        else:
            result = cv2.resize(scaled_result, (origin_w, origin_h))
            result = result.astype(np.uint8)

        os.remove(scaled_src_path)
        os.remove(scaled_mask_path)
        os.remove(scaled_inpainting_result_path)

        return result, dilated_scaled_mask
