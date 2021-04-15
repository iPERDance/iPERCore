# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
import numpy as np
import cv2
import os
from tqdm import tqdm

from iPERCore.tools.human_trackers import build_tracker
from iPERCore.tools.human_pose2d_estimators import build_pose2d_estimator
from iPERCore.tools.human_pose3d_estimators import build_pose3d_estimator, build_pose3d_refiner
from iPERCore.tools.human_mattors import build_mattor
from iPERCore.tools.background_inpaintors import build_background_inpaintors

from iPERCore.tools.processors.base_preprocessor import BaseProcessor
from iPERCore.tools.human_digitalizer.renders import SMPLRenderer

from iPERCore.services.options.process_info import ProcessInfo


class Preprocessor(BaseProcessor):

    def __init__(self, cfg, proc_size=512, device=torch.device("cuda:0")):
        """

        Args:
            cfg: the configurations for Preprocessor, it comes from the followings toml file,
                [Preprocess]
                    ## The configuration of Preprocessing.

                    # Set the max number of Preprocessor Instance for each GPU.
                    MAX_PER_GPU_PROCESS =  1

                    # Filter the invalid 2D kps.
                    filter_invalid = true

                    # 2D and 3D pose temporal smoooth.
                    temporal = true

                    [Preprocess.Cropper]
                        # The configurations of Image Cropper
                        src_crop_factor = 1.3
                        ref_crop_factor = 3.0

                    [Preprocess.Tracker]
                        # The configurations of Human Tracker, currently, it only supports the most naive `max_box` tracker，
                        # which chooses the large bounding-box of each image.
                        tracker_name = "max_box"

                    [Preprocess.Pose2dEstimator]
                        # The configurations of Human 2D Pose Estimation, currently, it only supports the `openpose` estimator.
                        pose2d_name = "openpose"
                        pose2d_cfg_path = "./assets/configs/pose2d/openpose/body25.toml"

                    [Preprocess.Pose3dEstimator]
                        # The configurations of Human 3D Pose Estimation, currently, it only supports the `spin` estimator.
                        pose3d_name = "spin"
                        pose3d_cfg_path = "./assets/configs/pose3d/spin.toml"

                        use_smplify = true
                        smplify_name = "smplify"
                        smplify_cfg_path = "./assets/configs/pose3d/smplify.toml"
                        use_lfbgs = true

                    [Preprocess.HumanMattors]
                        # The configurations of HumanMattors.
                        mattor_name = "point_render+gca"
                        mattor_cfg_path = "./assets/configs/mattors/point_render+gca.toml"

                    [Preprocess.BackgroundInpaintor]
                        # The configurations of BackgrounInpaintor.
                        inpaintor_name = "mmedit_inpainting"
                        inpaintor_cfg_path = "./assets/configs/inpaintors/mmedit_inpainting.toml"

            proc_size (int): the processed image size.

            device (torch.device):
        """

        super().__init__()

        # build the tracker
        tracker = build_tracker(name=cfg.Preprocess.Tracker.tracker_name)

        # build the pose2d estimator
        self.pose2d_estimator = build_pose2d_estimator(
            name=cfg.Preprocess.Pose2dEstimator.name,
            cfg_or_path=cfg.Preprocess.Pose2dEstimator.cfg_path,
            tracker=tracker,
            device=device
        )

        # build the pose3d estimator
        self.pose3d_estimator = build_pose3d_estimator(
            name=cfg.Preprocess.Pose3dEstimator.name,
            cfg_or_path=cfg.Preprocess.Pose3dEstimator.cfg_path,
            device=device
        )

        # build the pose3d refiner
        if cfg.Preprocess.use_smplify:
            self.pose3d_refiner = build_pose3d_refiner(
                name=cfg.Preprocess.Pose3dRefiner.name,
                cfg_or_path=cfg.Preprocess.Pose3dRefiner.cfg_path,
                use_lbfgs=cfg.Preprocess.Pose3dRefiner.use_lfbgs,
                joint_type=cfg.Preprocess.Pose2dEstimator.joint_type,
                device=device
            )
        else:
            self.pose3d_refiner = None

        # build the human mattor
        self.human_parser = build_mattor(
            name=cfg.Preprocess.HumanMattors.name,
            cfg_or_path=cfg.Preprocess.HumanMattors.cfg_path,
            device=device
        )

        self.inpaintor = build_background_inpaintors(
            name=cfg.Preprocess.BackgroundInpaintor.name,
            cfg_or_path=cfg.Preprocess.BackgroundInpaintor.cfg_path,
            device=device
        )

        self.render = SMPLRenderer(
            face_path=cfg.face_path,
            fim_enc_path=cfg.fim_enc_path,
            uv_map_path=cfg.uv_map_path,
            part_path=cfg.part_path,
            map_name=cfg.map_name,
            image_size=proc_size, fill_back=False, anti_aliasing=True,
            background_color=(0, 0, 0), has_front=True, top_k=3
        ).to(device)
        self.proc_size = proc_size
        self.device = device
        self.cfg = cfg

    def run_detector(self, image: np.ndarray):
        """

        Args:
            image (np.ndarray): it must be (height, width, 3) with np.uint8 in the range of [0, 255], BGR channel.
        Returns:
            output (dict):
        """

        # print(image_or_path)
        output = self.pose2d_estimator.run_single_image(image)
        return output

    def run_inpaintor(self, img_path, mask_path, dilate_kernel_size=19, dilate_iter_num=3):
        """

        Args:
            img_path (str): the full image path;
            mask_path (str): the mask path, 0 means the background, 255 means the area need to be inpainted;
            dilate_kernel_size (int):
            dilate_iter_num (int):

        Returns:
            bg_img (np.ndarray): inpainted background image, (h, w, 3), is in the range of [0, 255] with BGR channel.

        """

        bg_img, _ = self.inpaintor.run_inpainting(
            img_path, mask_path,
            dilate_kernel_size=dilate_kernel_size, dilate_iter_num=dilate_iter_num
        )

        return bg_img

    def _execute_post_pose3d(self, processed_info: ProcessInfo, use_smplify: bool = True,
                             filter_invalid: bool = True, temporal: bool = True):

        out_img_dir = processed_info["out_img_dir"]

        valid_img_info = processed_info["valid_img_info"]
        valid_img_names = valid_img_info["names"]
        valid_ids = valid_img_info["ids"]
        all_img_paths = [os.path.join(out_img_dir, name) for name in valid_img_names]

        crop_boxes_XYXY = processed_info["processed_cropper"]["crop_boxes_XYXY"]
        crop_keypoints = processed_info["processed_cropper"]["crop_keypoints"]

        if use_smplify:
            outputs = self.pose3d_estimator.run_with_smplify(
                all_img_paths, crop_boxes_XYXY, crop_keypoints, self.pose3d_refiner,
                batch_size=self.cfg.Preprocess.Pose3dEstimator.batch_size,
                num_workers=self.cfg.Preprocess.Pose3dEstimator.num_workers,
                filter_invalid=filter_invalid, temporal=temporal
            )

            all_init_smpls = outputs["all_init_smpls"]
            all_opt_smpls = outputs["all_opt_smpls"]
            all_valid_ids = outputs["all_valid_ids"]

            all_init_smpls = all_init_smpls.numpy()
            all_opt_smpls = all_opt_smpls.numpy()

        else:
            outputs = self.pose3d_estimator.run(
                all_img_paths, crop_boxes_XYXY,
                batch_size=self.cfg.Preprocess.Pose3dEstimator.batch_size,
                num_workers=self.cfg.Preprocess.Pose3dEstimator.num_workers,
                filter_invalid=filter_invalid
            )

            all_init_smpls = outputs["all_init_smpls"]
            all_valid_ids = outputs["all_valid_ids"]

            all_init_smpls = all_init_smpls.numpy()
            all_opt_smpls = all_init_smpls

        smpls_results = {
            "cams": all_opt_smpls[:, 0:3],
            "pose": all_opt_smpls[:, 3:-10],
            "shape": all_opt_smpls[:, -10:],
            "init_pose": all_init_smpls[:, 3:-10],
            "init_shape": all_init_smpls[:, -10:],
        }

        # update processed_pose3d
        processed_info["processed_pose3d"] = smpls_results

        # update valid_img_info
        valid_img_info["names"] = [valid_img_names[i] for i in all_valid_ids]
        valid_img_info["ids"] = [valid_ids[i] for i in all_valid_ids]
        valid_img_info["pose3d_ids"] = all_valid_ids.tolist()
        valid_img_info["stage"] = "pose3d"

        processed_info["valid_img_info"] = valid_img_info

        processed_info["has_run_3dpose"] = True

    def _execute_post_parser(self, processed_info: ProcessInfo):
        out_img_dir = processed_info["out_img_dir"]
        out_parse_dir = processed_info["out_parse_dir"]

        valid_img_info = processed_info["valid_img_info"]
        valid_img_names = valid_img_info["names"]
        valid_ids = valid_img_info["ids"]

        parser_valid_ids, mask_outs, alpha_outs = self.human_parser.run(
            out_img_dir, out_parse_dir, valid_img_names, save_visual=False
        )

        # update the valid_img_info
        valid_img_info["names"] = [valid_img_names[i] for i in parser_valid_ids]
        valid_img_info["ids"] = [valid_ids[i] for i in parser_valid_ids]
        valid_img_info["parse_ids"] = parser_valid_ids
        valid_img_info["stage"] = "parser"
        processed_info["valid_img_info"] = valid_img_info

        # add to 'processed_parse'
        processed_info["has_run_parser"] = True

    def _execute_post_find_front(self, processed_info: ProcessInfo, num_candidate=25, render_size=256):
        from iPERCore.tools.utils.geometry import mesh

        def comp_key(pair):
            return pair[0] + pair[1]

        processed_pose3d = processed_info["processed_pose3d"]
        cams = processed_pose3d["cams"]
        pose = processed_pose3d["pose"]
        shape = processed_pose3d["shape"]

        valid_img_info = processed_info["valid_img_info"]
        valid_img_names = valid_img_info["names"]

        length = len(valid_img_names)

        device = self.device
        render = SMPLRenderer(image_size=render_size).to(device)

        body_ids = set(mesh.get_part_face_ids(
            part_type="body_front",
            mapping_path=self.cfg.fim_enc_path,
            part_path=self.cfg.part_path,
            front_path=self.cfg.front_path,
            head_path=self.cfg.head_path,
            facial_path=self.cfg.facial_path
        ))
        face_ids = set(mesh.get_part_face_ids(
            part_type="head_front",
            mapping_path=self.cfg.fim_enc_path,
            part_path=self.cfg.part_path,
            front_path=self.cfg.front_path,
            head_path=self.cfg.head_path,
            facial_path=self.cfg.facial_path
        ))

        front_counts = []  # [(body_cnt, face_cnt, ids), (body_cnt, face_cnt, ids), ...]

        CANDIDATE = min(num_candidate, length)
        for i in tqdm(range(length)):
            _cams = torch.tensor(cams[i:i + 1]).to(device)
            _poses = torch.tensor(pose[i:i + 1]).to(device)
            _shapes = torch.tensor(shape[i:i + 1]).to(device)

            with torch.no_grad():
                _verts, _, _ = self.pose3d_estimator.body_model(beta=_shapes, theta=_poses, get_skin=True)

            _fim = set(render.render_fim(_cams, _verts).long()[0].unique()[1:].cpu().numpy())

            bd_cnt = len(body_ids & _fim)
            fa_cnt = len(face_ids & _fim)

            front_counts.append((bd_cnt, fa_cnt, i))

        front_counts.sort(key=comp_key, reverse=True)
        ft_candidates = front_counts[0:CANDIDATE]
        bk_candidates = list(reversed(front_counts[-CANDIDATE:]))

        video_front_counts = {
            "ft": {
                "body_num": [int(pair[0]) for pair in ft_candidates],
                "face_num": [int(pair[1]) for pair in ft_candidates],
                "ids": [pair[2] for pair in ft_candidates]
            },
            "bk": {
                "body_num": [int(pair[0]) for pair in bk_candidates],
                "face_num": [int(pair[1]) for pair in bk_candidates],
                "ids": [pair[2] for pair in bk_candidates]
            }
        }
        # print("ft_candidates", ft_candidates)
        # print("bk_candidates", bk_candidates)

        # add to 'processed_front_info'
        processed_info["processed_front_info"] = video_front_counts
        processed_info["has_find_front"] = True

    def _execute_post_inpaintor(self, processed_info: ProcessInfo,
                                dilate_kernel_size: int = 19, dilate_iter_num: int = 3,
                                bg_replace: bool = False):

        out_img_dir = processed_info["out_img_dir"]
        out_parse_dir = processed_info["out_parse_dir"]
        out_bg_dir = processed_info["out_bg_dir"]

        front_info = processed_info["processed_front_info"]
        front_ids = front_info["ft"]["ids"]
        back_ids = front_info["bk"]["ids"]
        src_ids = set(front_ids) | set(back_ids)

        valid_img_names = processed_info["valid_img_info"]["names"]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))

        for i in tqdm(src_ids):
            img_name = valid_img_names[i]
            img_path = os.path.join(out_img_dir, img_name)
            image = cv2.imread(img_path)

            # TODO, only works for image name with "xxx.png" or "xxx.jpg"
            name = str(img_name.split(".")[0])
            msk_path = os.path.join(out_parse_dir, name + "_mask.png")
            mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

            bg_img = self.run_inpaintor(
                img_path, mask, dilate_kernel_size=dilate_kernel_size,
                dilate_iter_num=dilate_iter_num
            )

            inpainted_path = os.path.join(out_bg_dir, name + "_inpainted.png")
            cv2.imwrite(inpainted_path, bg_img)

            if bg_replace:
                mask = cv2.dilate(mask, kernel, iterations=1)
                bg_area = (mask == 0)
                bg_img[bg_area, :] = image[bg_area, :]
                replaced_path = os.path.join(out_bg_dir, name + "_replaced.png")
                cv2.imwrite(replaced_path, bg_img)

        processed_info["processed_background"]["replace"] = bg_replace
        processed_info["has_run_inpaintor"] = True

    def _save_visual(self, processed_info: ProcessInfo):
        from iPERCore.tools.utils.visualizers.smpl_visualizer import visual_pose3d_results

        out_img_dir = processed_info["out_img_dir"]
        out_parse_dir = processed_info["out_parse_dir"]
        out_visual_path = processed_info["out_visual_path"]

        processed_cropper = processed_info["processed_cropper"]
        processed_pose3d = processed_info["processed_pose3d"]
        valid_img_info = processed_info["valid_img_info"]

        valid_img_names = valid_img_info["names"]
        crop_ids = valid_img_info["crop_ids"]
        pose3d_ids = valid_img_info["pose3d_ids"]
        parse_ids = valid_img_info["parse_ids"]

        crop_keypoints = processed_cropper["crop_keypoints"]

        if len(crop_keypoints) > 0:
            pose3d_to_parse_ids = [pose3d_ids[ids] for ids in parse_ids]
            crop_to_pose3d_ids = [crop_ids[ids] for ids in pose3d_to_parse_ids]
            all_keypoints = [crop_keypoints[ids]["pose_keypoints_2d"] for ids in crop_to_pose3d_ids]
        else:
            all_keypoints = None

        prepare_smpls_info = {
            "all_init_smpls": np.concatenate([processed_pose3d["cams"][parse_ids],
                                              processed_pose3d["init_pose"][parse_ids],
                                              processed_pose3d["init_shape"][parse_ids]], axis=-1),

            "all_opt_smpls": np.concatenate([processed_pose3d["cams"][parse_ids],
                                             processed_pose3d["pose"][parse_ids],
                                             processed_pose3d["shape"][parse_ids]], axis=-1),

            "valid_img_names": valid_img_names,
            "all_keypoints": all_keypoints
        }

        visual_pose3d_results(
            out_visual_path, out_img_dir, prepare_smpls_info, parse_dir=out_parse_dir,
            smpl_model=self.cfg.smpl_model, image_size=self.proc_size, fps=25,

        )

    def close(self):
        pass

