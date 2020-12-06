import os
import cv2
from tqdm import tqdm
from typing import Dict, Any, Union, List


def crop_func(src_img_dir, out_img_dir, image_name, crop_size, fmt_active_boxes, keypoints, boxes_XYXY):
    """

    Args:
        src_img_dir:
        out_img_dir:
        image_name:
        crop_size:
        fmt_active_boxes:
        keypoints:
        boxes_XYXY:

    Returns:

    """

    image_path = os.path.join(src_img_dir, image_name)
    orig_img = cv2.imread(image_path)

    crop_info = self.process_crop_img(orig_img, fmt_active_boxes, crop_size)

    if crop_info is not None:
        cv2.imwrite(os.path.join(out_img_dir, image_name), crop_info["image"])

        if has_keypoints:
            scale = crop_info["scale"]
            start_pt = crop_info["start_pt"]

            orig_kps = keypoints[i]
            crop_kps = self.crop_resize_kps(orig_kps, scale, start_pt)

            orig_boxes = boxes_XYXY[i]
            crop_boxes = self.crop_resize_boxes(orig_boxes, scale, start_pt)

            self._update_context(context, "crop_keypoints", crop_kps, is_update=True)
            self._update_context(context, "crop_boxes_XYXY", crop_boxes, is_update=True)


def _execute_crop_img(self, context: Dict[str, Any], crop_size: int, bg_path: Union[str, None] = None):
    src_img_dir = context["src_img_dir"]
    out_img_dir = context["out_img_dir"]
    out_bg_dir = context["out_bg_dir"]
    crop_img_names = context["update"]["crop_img_names"]

    fmt_active_boxes = context["active_boxes_XYXY"]
    keypoints = context["update"]["keypoints"]
    boxes_XYXY = context["update"]["boxes_XYXY"]
    has_keypoints = len(keypoints) > 0

    for i, image_name in enumerate(tqdm(crop_img_names)):
        image_path = os.path.join(src_img_dir, image_name)
        orig_img = cv2.imread(image_path)

        crop_info = self.process_crop_img(orig_img, fmt_active_boxes, crop_size)

        if crop_info is not None:
            cv2.imwrite(os.path.join(out_img_dir, image_name), crop_info["image"])

            if has_keypoints:
                scale = crop_info["scale"]
                start_pt = crop_info["start_pt"]

                orig_kps = keypoints[i]
                crop_kps = self.crop_resize_kps(orig_kps, scale, start_pt)

                orig_boxes = boxes_XYXY[i]
                crop_boxes = self.crop_resize_boxes(orig_boxes, scale, start_pt)

                self._update_context(context, "crop_keypoints", crop_kps, is_update=True)
                self._update_context(context, "crop_boxes_XYXY", crop_boxes, is_update=True)

                # renorm_orig_kps = orig_kps["pose_keypoints_2d"].copy()
                # orig_skeleton = draw_skeleton(orig_img, renorm_orig_kps)
                #
                # renorm_crop_kps = crop_kps["pose_keypoints_2d"].copy()
                # crop_skeleton = draw_skeleton(crop_info["image"], renorm_crop_kps)
                #
                # orig_skeleton = orig_skeleton[None]
                # crop_skeleton = crop_skeleton[None]
                #
                # ox0, oy0, ox1, oy1 = [int(x) for x in orig_boxes]
                # cx0, cy0, cx1, cy1 = [int(x) for x in crop_boxes]
                #
                # orig_crop = orig_skeleton[:, :, oy0:oy1, ox0:ox1]
                # crop_crop = crop_skeleton[:, :, cy0:cy1, cx0:cx1]
                #
                # visualizer.vis_named_img("orig", orig_skeleton, denormalize=False, transpose=False)
                # visualizer.vis_named_img("crop", crop_skeleton, denormalize=False, transpose=False)
                #
                # visualizer.vis_named_img("crop_orig", orig_crop, denormalize=False, transpose=False)
                # visualizer.vis_named_img("crop_crop", crop_crop, denormalize=False, transpose=False)

    if (bg_path is not None) and (bg_path != "None"):
        bg_img = cv2.imread(bg_path)
        bg_name = os.path.split(bg_path)[-1]
        crop_info = self.process_crop_img(bg_img, fmt_active_boxes, crop_size)
        cv2.imwrite(os.path.join(out_bg_dir, bg_name), crop_info["image"])

    # finish crop images
    self._update_context(context, "has_run_cropper", True)
