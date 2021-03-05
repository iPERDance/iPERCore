# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import time
import torch
import cv2
import numpy as np
import os
from easydict import EasyDict
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from mmedit.apis import init_model, matting_inference

from iPERCore.tools.utils.filesio.cv_utils import compute_scaled_size
from iPERCore.tools.utils.filesio.persistence import load_toml_file


class PointRenderGCAMattor(object):

    def __init__(self, cfg_or_path, device=torch.device("cuda:0")):

        """

        Args:
            cfg_or_path: the config object, it contains the following information:
                seg_cfg_path="./assets/configs/detection/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco.py",
                seg_ckpt_path="./assets/checkpoints/detection/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth",
                matting_cfg_path="./assets/configs/editing/mattors/gca/gca_r34_4x10_200k_comp1k.py",
                matting_ckpt_path="./assets/checkpoints/mattors/gca_r34_4x10_200k_comp1k_SAD-34.77_20200604_213848-4369bea0.pth",
                person_label_index = 0

                temp_dir="./assets/temp"

                trimap_control_size = 300
                matting_image_size = 512
                morph_kernel_size = 3
                erode_iter_num = 2
                dilate_iter_num = 7

            device:
        """

        if isinstance(cfg_or_path, str):
            cfg = EasyDict(load_toml_file(cfg_or_path))
        else:
            cfg = cfg_or_path

        self.trimap_control_size = cfg.trimap_control_size
        self.matting_image_size = cfg.matting_image_size

        self.erode_iter_num = cfg.erode_iter_num
        self.dilate_iter_num = cfg.dilate_iter_num
        self.morph_kernel_size = cfg.morph_kernel_size

        """ point_rend_r50_caffe_fpn_mstrain_3x_coco  """
        self.detection_config_file = cfg.seg_cfg_path
        self.detection_checkpoint_file = cfg.seg_ckpt_path
        self.person_label_index = cfg.person_label_index

        """ gca_r34_4x10_200k_comp1k """
        self.editing_config_file = cfg.matting_cfg_path
        self.editing_checkpoint_file = cfg.matting_ckpt_path

        self.device = device
        self.detection_model = init_detector(self.detection_config_file, self.detection_checkpoint_file, device=device)
        self.matting_model = init_model(self.editing_config_file, self.editing_checkpoint_file,
                                        device=device.__str__())

        self.temp_dir = cfg.temp_dir

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def generate_trimap(self, mask):
        """

        Args:
            mask (np.ndarray): (h, w) 0 or 1

        Returns:
            trimap (np.ndarray): (h, w) is in the range [0, 255]
        """

        origin_h, origin_w = mask.shape
        scaled_size = compute_scaled_size((origin_w, origin_h), control_size=self.trimap_control_size)

        # scale to control size
        mask = cv2.resize(mask, scaled_size, interpolation=cv2.INTER_NEAREST)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))

        inner = cv2.erode(mask, kernel, iterations=self.erode_iter_num)
        outer = cv2.dilate(mask, kernel, iterations=self.dilate_iter_num)
        trimap = inner * 255 + (outer - inner) * 128
        trimap = cv2.resize(trimap, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)

        return trimap

    def run_detection(self, img_path):
        """
            Run detection to get the segmentation mask and trimap, assuming that there is only a single human
        in the image.

        Args:
            img_path (str): the image path

        Returns:
            has_person (bool): whether there is person or not.
            segm_mask (np.ndarray): (h, w)
            trimap (np.ndarray): (h, w)
        """
        result = inference_detector(self.detection_model, img_path)
        bbox_result, segm_result = result
        num_people = len(bbox_result[0])

        has_person = num_people > 0
        if has_person:
            # (src_h, src_w) 0 or 1
            # in COCO dataset, `0` represents the person,
            # segm_result[self.person_label_index] represents all the results of Person,
            # segm_result[self.person_label_index][0] represents the first Person segmentation result.
            segm_mask = segm_result[self.person_label_index][0].astype(np.float32)

            # (src_h, src_w) 0 or 128 or 255
            trimap = self.generate_trimap(segm_mask)
        else:
            segm_mask = []
            trimap = []

        return has_person, segm_mask, trimap

    def run_matting(self, img_or_path):
        """
        1. run instance segmentation with PointRender, detection first;
        2. generate trimap;
        3. run matting;

        Args:
            img_or_path (str or np.ndarray): (h, w, 3) is in the range of [0, 255] with BGR channel space.

        Returns:
            has_person (bool): whether there is person or not.
            segm_mask (np.ndarray): (h, w), 0 or 1
            trimap (np.ndarray): (h, w), 0 or 128, or 255;
            pred_alpha (np.ndarray): (h, w), is in the range of [0, 1], np.float32
        """

        # TODO, do not write the middle outputs to disk, and make them in memory.
        #  scaled_src_path, scaled_trimap_path

        # img_name = str(time.time())
        # img_path = os.path.join(self.temp_dir, img_name)

        path = os.path.normpath(img_or_path)
        img_name = path.replace(os.sep, "_")
        img_path = os.path.join(self.temp_dir, img_name)

        if isinstance(img_or_path, str):
            src_img = cv2.imread(img_or_path)
        else:
            src_img = img_or_path.copy()

        # 1. run detection, instance segmentation and generate trimap
        has_person, segm_mask, trimap = self.run_detection(img_or_path)
        pred_alpha = []

        if has_person:
            # 2. run matting algorithm
            scaled_src_path = img_path + '.matting.png'
            scaled_trimap_path = img_path + '.trimap.png'

            origin_h, origin_w = src_img.shape[:2]
            scaled_size = compute_scaled_size((origin_w, origin_h), control_size=self.matting_image_size)
            scaled_src_img = cv2.resize(src_img, scaled_size)
            scaled_trimap = cv2.resize(trimap, scaled_size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(scaled_src_path, scaled_src_img)
            cv2.imwrite(scaled_trimap_path, scaled_trimap)

            # (scaled_h, scaled_w) [0, 1]
            pred_alpha = matting_inference(self.matting_model, scaled_src_path, scaled_trimap_path)

            # (origin_h, origin_w) [0, 1]
            pred_alpha = cv2.resize(pred_alpha, (origin_w, origin_h))

            os.remove(scaled_src_path)
            os.remove(scaled_trimap_path)

        return has_person, segm_mask, trimap, pred_alpha

    def run(self, src_dir, out_dir, src_img_names=None, save_visual=True):
        """
        Run human matting of all the images on a directory.

        Args:
            src_dir (str):
            out_dir (str):
            src_img_names (List[str]):
            save_visual (bool):

        Returns:
            None
        """

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        all_img_names = os.listdir(src_dir)
        all_img_names.sort()

        if src_img_names is None:
            processed_img_names = all_img_names
        else:
            processed_img_names = []
            for img_name in src_img_names:
                processed_img_names.append(img_name)

        mask_outs = []
        alpha_outs = []
        valid_ids = []
        for ids, img_name in enumerate(tqdm(processed_img_names)):
            img_path = os.path.join(src_dir, img_name)
            has_person, segm_mask, trimap, pred_alpha = self.run_matting(img_path)

            if has_person:
                valid_ids.append(ids)

                name = img_name.split('.')[0]
                mask_path = os.path.join(out_dir, name + "_mask.png")
                alpha_path = os.path.join(out_dir, name + "_alpha.png")

                cv2.imwrite(alpha_path, (pred_alpha * 255).astype(np.uint8))
                cv2.imwrite(mask_path, (segm_mask * 255).astype(np.uint8))

                mask_outs.append(mask_path)
                alpha_outs.append(alpha_path)

                if save_visual:
                    cv2.imwrite(os.path.join(out_dir, name + "trimap.png"), trimap)

        return valid_ids, mask_outs, alpha_outs
