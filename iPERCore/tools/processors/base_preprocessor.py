# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import abc
import numpy as np
import cv2
import os
from enum import Enum, unique
from typing import Union
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor

from iPERCore.tools.processors import process_utils
from iPERCore.services.options.process_info import ProcessInfo


class BaseProcessor(metaclass=abc.ABCMeta):
    """
    Consumer for preprocessing, it contains the following steps:
    1. It firstly use the human detector to crop the bounding boxes of the person;
    2. then, it center crops a square image from the original image, and it might use pad and resize;
    3. next, it will estimate the 3D cam, pose, and shape of the 3D parametric model (SMPL);
    4. then, it will sort the images by counting the number of front visible triangulated faces;
    5. finally, it will run the human matting algorithm to get the mask of human;

    See self.execute() for more details.
    """

    @unique
    class ACTIONS(Enum):
        CROP_FOR_RENDER = 0
        CROP_FOR_HMR = 1
        CROP_FOR_SPIN = 2

    @property
    def valid_actions(self):
        return [self.ACTIONS.CROP_FOR_RENDER, self.ACTIONS.CROP_FOR_HMR, self.ACTIONS.CROP_FOR_SPIN]

    def execute(self, processed_info: ProcessInfo,
                crop_size=512, estimate_boxes_first=True, factor=1.25, num_workers=0,
                use_simplify=True, filter_invalid=True, parser=True,
                inpaintor=False, dilate_kernel_size=19, dilate_iter_num=3, bg_replace=False,
                find_front=True, num_candidate=25, render_size=256,
                temporal=True, visual=True):
        """

        Args:
            processed_info:
            crop_size:
            estimate_boxes_first:
            factor:
            num_workers:
            use_simplify:
            filter_invalid:
            parser:
            inpaintor:
            dilate_kernel_size:
            dilate_iter_num:
            bg_replace:
            find_front:
            num_candidate:
            render_size:
            temporal:
            visual:

        Returns:

        """

        meta_input = processed_info["input_info"]["meta_input"]
        input_path = meta_input["path"]
        bg_path = meta_input["bg_path"]

        src_img_dir = process_utils.format_imgs_dir(input_path, processed_info["src_img_dir"])
        out_img_dir = processed_info["out_img_dir"]

        # 1 (2). run detector to crop
        if estimate_boxes_first:
            if not processed_info["has_run_detector"]:
                # 1.1 run detector and selector to find the active boxes
                print(f"\t1.1 Preprocessing, running {self.__class__.__name__} "
                      f"to detect the human boxes of {src_img_dir}...")
                self._execute_detector(processed_info, factor=factor)
                print(f"\t1.1 Preprocessing, finish detect the human boxes of {src_img_dir} ...")

            if not processed_info["has_run_cropper"]:
                # 1.2 crop image
                # check whether the image has been cropped or not
                print(f"\t1.2 Preprocessing, cropping all images in {src_img_dir} by estimated boxes ...")
                self._execute_crop_img(processed_info, crop_size, bg_path, num_workers=num_workers)
                print(f"\t1.2 Preprocessing, finish crop the human by boxes, and save them in {out_img_dir} ...")
        else:
            if not processed_info["has_run_cropper"]:
                # 1.1 (2) crop image
                print(f"\t1.2 Preprocessing, directly resize all images in {src_img_dir} by estimated boxes....")
                self._direct_resize_img(processed_info, crop_size, bg_path)
                print(f"\t1.2 Preprocessing, finish crop the human by boxes, and save them in {out_img_dir} ...")

        processed_info.serialize()

        # 3. run smpl estimator
        if not processed_info["has_run_3dpose"]:
            print(f"\t1.3 Preprocessing, running {self.__class__.__name__} to 3D pose estimation of all images in"
                  f"{out_img_dir} ...")
            self._execute_post_pose3d(processed_info, use_simplify, filter_invalid=filter_invalid, temporal=temporal)
            print(f"\t1.3 Preprocessing, finish 3D pose estimation successfully ....")

        processed_info.serialize()

        # 4. run human parser
        if parser and not processed_info["has_run_parser"]:
            print(f"\t1.4 Preprocessing, running {self.__class__.__name__} to run human matting in "
                  f"{processed_info['out_parse_dir']} ... ")
            self._execute_post_parser(processed_info)
            print(f"\t1.4 Preprocessing, finish run human matting.")

        processed_info.serialize()

        # 5. find the front face images
        if find_front and not processed_info["has_find_front"]:
            print(f"\t1.5 Preprocessing, running {self.__class__.__name__} to find {num_candidate} "
                  f"candidates front images in {out_img_dir} ...")
            self._execute_post_find_front(processed_info, num_candidate, render_size)
            print(f"\t1.5 Preprocessing, finish find the front images ....")

        processed_info.serialize()

        # 6. run background inpaintor
        if inpaintor and not processed_info["has_run_inpaintor"]:
            print(f"\t1.6 Preprocessing, running {self.__class__.__name__} to run background inpainting ...")
            self._execute_post_inpaintor(
                processed_info,
                dilate_kernel_size=dilate_kernel_size,
                dilate_iter_num=dilate_iter_num,
                bg_replace=bg_replace
            )
            print(f"\t1.6 Preprocessing, finish run background inpainting ....")

        processed_info.serialize()

        if visual and not os.path.exists(processed_info["out_visual_path"]):
            print(f"\t1.7 Preprocessing, saving visualization to {processed_info['out_visual_path']} ...")
            self._save_visual(processed_info)
            print(f"\t1.7 Preprocessing, saving visualization to {processed_info['out_visual_path']} ...")

        processed_info["has_finished"] = True
        processed_info.serialize()

        print("{} has finished...".format(self.__class__.__name__))

    @abc.abstractmethod
    def run_detector(self, *args, **kwargs):
        """

        Args:
            *args:
            **kwargs:

        Returns:
            result (dict): a dictionary must contains the following items:
                --has_person (bool): the flag indicates there is person detected in the input image or not;
                --boxes_XYXY (tuple, list or np.narray): the bounding boxes, (x0, y0, x1, y1),
                --orig_shape (tuple): (height, width), the shape of input image.
        """
        pass

    def _execute_detector(self, processed_info: ProcessInfo, factor: float):
        src_img_dir = processed_info["src_img_dir"]
        images_names = os.listdir(src_img_dir)

        # TODO: Be careful to the image names when listing the image folder in video sequences.
        images_names.sort()
        # print(images_names)

        active_boxes = None
        orig_shape = None
        all_boxes_XYXY = []
        all_keypoints = []

        all_pose2d_img_ids = []
        all_pose2d_img_names = []
        for i, image_name in enumerate(tqdm(images_names)):
            image_path = os.path.join(src_img_dir, image_name)
            image = cv2.imread(image_path)
            cur_shape = image.shape[0:2]

            if orig_shape is not None and orig_shape != cur_shape:
                warnings.warn(f"{image_path} has an image shape {cur_shape}, but the previous image "
                              f"has the different image shape {orig_shape}, here we resize the current image.")

                image = cv2.resize(image, (orig_shape[1], orig_shape[0]))

            result = self.run_detector(image)

            if orig_shape is None:
                orig_shape = result["orig_shape"]

            if result["has_person"]:
                boxes_XYXY = result["boxes_XYXY"]
                active_boxes = process_utils.update_active_boxes(boxes_XYXY, active_boxes)

                all_boxes_XYXY.append(boxes_XYXY)
                all_pose2d_img_ids.append(i)
                all_pose2d_img_names.append(image_name)

                if "keypoints" in result:
                    all_keypoints.append(result["keypoints"])
            else:
                warnings.warn(f"there is no person in {image_name}, and this frame will be ignored.")

        if factor == 0:
            max_size = max(orig_shape[1], orig_shape[0])
            cx = orig_shape[1] // 2
            cy = orig_shape[0] // 2
            half = max_size // 2
            fmt_active_boxes = (max(0, cx - half), max(0, cy - half),
                                min(cx + half, orig_shape[1]), min(cy + half, orig_shape[0]))
        else:
            fmt_active_boxes = process_utils.fmt_active_boxes(active_boxes, orig_shape, factor=factor)

        # 1. add 'orig_shape' to processed_info
        processed_info["orig_shape"] = orig_shape

        # 2. add 'processed_pose2d' to processed_info
        processed_pose2d = {
            "boxes_XYXY": all_boxes_XYXY,
            "keypoints": all_keypoints
        }
        processed_info["processed_pose2d"] = processed_pose2d
        processed_info["processed_cropper"]["active_boxes_XYXY"] = fmt_active_boxes

        # 3. add 'valid_img_info' to processed_info
        valid_img_info = {
            "names": all_pose2d_img_names,
            "ids": all_pose2d_img_ids,
            "stage": "pose2d"
        }
        processed_info["valid_img_info"] = valid_img_info

        # finish detection
        processed_info["has_run_detector"] = True

    def _execute_crop_img(self, processed_info: ProcessInfo,
                          crop_size: int, bg_path: Union[str, None] = None,
                          num_workers: int = 0):
        src_img_dir = processed_info["src_img_dir"]
        out_img_dir = processed_info["out_img_dir"]
        orig_shape = processed_info["orig_shape"]

        valid_img_info = processed_info["valid_img_info"]
        valid_img_names = valid_img_info["names"]
        valid_ids = valid_img_info["ids"]

        processed_pose2d = processed_info["processed_pose2d"]

        all_boxes_XYXY = processed_pose2d["boxes_XYXY"]
        all_keypoints = processed_pose2d["keypoints"]
        has_keypoints = len(all_keypoints) > 0

        processed_cropper = processed_info["processed_cropper"]
        fmt_active_boxes = processed_cropper["active_boxes_XYXY"]

        all_crop_boxes_XYXY = []
        all_crop_keypoints = []

        if num_workers > 0:
            num_images = len(valid_img_names)
            src_paths = [os.path.join(src_img_dir, img_name) for img_name in valid_img_names]
            out_paths = [os.path.join(out_img_dir, img_name) for img_name in valid_img_names]
            crop_list = [crop_size] * num_images
            orig_shape_list = [orig_shape] * num_images
            fmt_active_boxes_list = [fmt_active_boxes] * num_images

            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                for (crop_boxes, crop_kps) in tqdm(pool.map(process_utils.crop_func,
                                                            src_paths, out_paths, crop_list,
                                                            orig_shape_list, fmt_active_boxes_list,
                                                            all_boxes_XYXY, all_keypoints)):

                    all_crop_boxes_XYXY.append(crop_boxes)

                    if crop_kps is not None:
                        all_crop_keypoints.append(crop_kps)

        else:
            for i, image_name in enumerate(tqdm(valid_img_names)):
                boxes_XYXY = all_boxes_XYXY[i]
                keypoints = all_keypoints[i] if has_keypoints else None

                src_path = os.path.join(src_img_dir, image_name)
                out_path = os.path.join(out_img_dir, image_name)

                crop_boxes, crop_kps = process_utils.crop_func(
                    src_path, out_path, crop_size, orig_shape,
                    fmt_active_boxes, boxes_XYXY, keypoints
                )
                all_crop_boxes_XYXY.append(crop_boxes)

                if crop_kps is not None:
                    all_crop_keypoints.append(crop_kps)

        if bg_path:
            out_actual_bg_dir = processed_info["out_actual_bg_dir"]

            bg_img = cv2.imread(bg_path)
            bg_name = os.path.split(bg_path)[-1]
            crop_info = process_utils.process_crop_img(bg_img, fmt_active_boxes, crop_size)
            cv2.imwrite(os.path.join(out_actual_bg_dir, bg_name), crop_info["image"])

        # finish crop images
        processed_info["valid_img_info"]["crop_ids"] = valid_ids
        processed_info["valid_img_info"]["stage"] = "crop"

        processed_info["processed_cropper"]["crop_boxes_XYXY"] = all_crop_boxes_XYXY
        processed_info["processed_cropper"]["crop_keypoints"] = all_crop_keypoints
        processed_info["has_run_cropper"] = True

    def _direct_resize_img(self, processed_info: ProcessInfo, crop_size=512, bg_path: Union[str, None] = None):
        src_img_dir = processed_info["src_img_dir"]
        out_img_dir = processed_info["out_img_dir"]
        out_bg_dir = processed_info["out_bg_dir"]
        images_names = os.listdir(src_img_dir)

        orig_shape = None

        all_crop_img_ids = []
        all_crop_img_names = []

        for i, image_name in enumerate(tqdm(images_names)):
            image_path = os.path.join(src_img_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (crop_size, crop_size))

            if orig_shape is None:
                orig_shape = image.shape[0:2]

            cv2.imwrite(os.path.join(out_img_dir, image_name), image)

            all_crop_img_ids.append(i)
            all_crop_img_names.append(image_name)

        if bg_path:
            out_actual_bg_dir = processed_info["out_actual_bg_dir"]

            bg_img = cv2.imread(bg_path)
            bg_img = cv2.resize(bg_img, (crop_size, crop_size))
            bg_name = os.path.split(bg_path)[-1]
            cv2.imwrite(os.path.join(out_actual_bg_dir, bg_name), bg_img)

        processed_info["processed_pose2d"]["active_boxes_XYXY"] = np.array([0, orig_shape[1], 0, orig_shape[0]])

        processed_info["valid_img_info"]["names"] = all_crop_img_names
        processed_info["valid_img_info"]["ids"] = all_crop_img_ids
        processed_info["valid_img_info"]["stage"] = "crop"

        processed_info["has_run_detector"] = True
        processed_info["has_run_cropper"] = True

    @abc.abstractmethod
    def _execute_post_pose3d(self, processed_info: ProcessInfo, use_simplify: bool = True,
                             filter_invalid: bool = True, temporal: bool = True):
        """

        Args:
            processed_info (ProcessInfo):
            use_simplify (bool): the flag to control use simplify to refine the estimated SMPL parameters
                by network or not.
            filter_invalid (bool): the flag to control whether to filter the valid or invalid estimated SMPL.
            temporal (bool):

        Returns:

        """
        pass

    @abc.abstractmethod
    def _execute_post_find_front(self, processed_info: ProcessInfo, num_candidate: int = 25, render_size: int = 256):
        """

        Args:
            processed_info (ProcessInfo):
            num_candidate (int):
            render_size (int):

        Returns:

        """
        pass

    @abc.abstractmethod
    def _execute_post_parser(self, processed_info: ProcessInfo):
        """

        Args:
            processed_info (ProcessInfo):

        Returns:

        """
        pass

    @abc.abstractmethod
    def _execute_post_inpaintor(self, processed_info: ProcessInfo, dilate_kernel_size: int = 19,
                                dilate_iter_num: int = 3, bg_replace: bool = True):
        """

        Args:
            processed_info (ProcessInfo):
            dilate_kernel_size (int):
            dilate_iter_num (int):
            bg_replace (bool):
        Returns:

        """
        pass

    @abc.abstractmethod
    def _save_visual(self, processed_info: ProcessInfo):
        """

        Args:
            processed_info (ProcessInfo):

        Returns:

        """
        pass

    @abc.abstractmethod
    def close(self):
        pass
