# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import numpy as np
import cv2
from tqdm import tqdm

from ..multimedia.video import convert_avi_to_mp4


colors = {
    "pink": [197, 27, 125],  # L lower leg
    "Violet": [238, 130, 238],
    "DarkViolet": [148, 0, 211],
    "light_pink": [233, 163, 201],  # L upper leg
    "light_green": [161, 215, 106],  # L lower arm
    "green": [77, 146, 33],  # L upper arm
    "IndianRed": [205, 92, 92],
    "RosyBrown2": [238, 180, 180],
    "red": [215, 48, 39],  # head
    "light_red": [252, 146, 114],  # head
    "light_orange": [252, 141, 89],  # chest
    "DarkOrange2": [238, 118, 0],
    "purple": [118, 42, 131],  # R lower leg
    "BlueViolet": [138, 43, 226],
    "light_purple": [175, 141, 195],  # R upper
    "light_blue": [145, 191, 219],  # R lower arm
    "MediumSlateBlue": [123, 104, 238],
    "DarkSlateBlue": [72, 61, 139],
    "NavyBlue": [0, 0, 128],
    "LightSlateBlue": [132, 112, 255],
    "blue": [69, 117, 180],  # R upper arm
    "gray": [130, 130, 130],  #
    "YellowGreen": [154, 205, 50],
    "LightCoral": [240, 128, 128],
    "Aqua": [0, 255, 255],
    "chocolate": [210, 105, 30],
    "white": [255, 255, 255],  #
}

jcolors = [
    "light_red", "light_pink", "light_green", "red", "pink", "green",
    "light_orange", "light_purple", "light_blue", "DarkOrange2", "purple", "blue",
    "MediumSlateBlue", "YellowGreen", "LightCoral", "YellowGreen", "green", "LightSlateBlue", "MediumSlateBlue",
    "DarkSlateBlue", "DarkSlateBlue", "Violet", "BlueViolet", "NavyBlue", "RosyBrown2", "Aqua", "chocolate"
]

ecolors = {
    0: "IndianRed",
    1: "RosyBrown2",
    2: "light_pink",
    3: "pink",
    4: "Violet",
    5: "DarkViolet",
    6: "light_blue",
    7: "DarkSlateBlue",
    8: "LightSlateBlue",
    9: "NavyBlue",
    10: "MediumSlateBlue",
    11: "blue",
    12: "BlueViolet",
    13: "DarkSlateBlue",
    14: "purple",
    15: "Violet",
    16: "BlueViolet",
    17: "RosyBrown2",
    18: "light_green",
    19: "YellowGreen",
    20: "light_red",
    21: "light_pink",
    22: "light_green",
    23: "pink",
    24: "green",
    25: "chocolate",
    26: "Aqua"
}


def draw_skeleton(orig_img, joints, radius=6, transpose=True, threshold=0.25):
    global colors, ecolors, jcolors

    image = orig_img.copy()

    input_is_float = False

    is_valid = joints[:, 2] > threshold

    if np.issubdtype(image.dtype, np.float32):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it"s slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # coco-plus-19
    if joints.shape[0] == 19:
        parents = np.array([
            1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
        ])

    # coco-whole-body-23
    elif joints.shape[0] == 23:
        #    0, 1, 2, 3, 4, 5,  6, 7, 8, 9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
        #    1, 2, 3, 4, 5, 6,  7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
        parents = np.array([
          # -1, 1, 1, 2, 3, -1, 6, 6, 7, 8,  9,  6, 7, 12, 13, 14, 15, 20, 20, 16, 23, 23, 17
            -1, 0, 0, 1, 2, -1, 5, 5, 6, 7,  8,  5, 6, 11, 12, 13, 14, 19, 19, 15, 22, 22, 16
        ])

    # cmu-body-25
    elif joints.shape[0] == 25:
        parents = np.array([
            # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
            -1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 14, 19, 14, 11, 22, 11
        ])

    elif joints.shape[0] == 26:
        parents = np.array([
            # 0, 1, 2, 3, 4,  5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
             -1, 0, 0, 1, 2, 18, 18, 5, 6, 7,  8, 19, 19, 11, 12, 13, 14,  0, -1, 18, 24, 25, 15, 16, 15, 16
        ])

    for i in range(len(parents)):
        pc = joints[i]

        pa_id = parents[i]
        if pa_id >= 0 and is_valid[pa_id] and is_valid[i]:
            pa = joints[pa_id]
            cv2.line(image, (int(pc[0]), int(pc[1])), (int(pa[0]), int(pa[1])), colors[ecolors[i]], radius // 2)

        if is_valid[i]:
            cv2.circle(image, (int(pc[0]), int(pc[1])), radius, colors[jcolors[i]], -1)

    if transpose:
        image = np.transpose(image, (2, 0, 1))

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)
    return image


def visual_pose2d_results(save_video_path, img_paths, pose_2d_results, fps=25):
    """

    Args:
        save_video_path:
        img_paths:
        pose_2d_results:
        fps:

    Returns:

    """

    length = len(pose_2d_results)

    if length <= 0:
        return

    height, width = pose_2d_results[0]["orig_shape"]

    tmp_avi_video_path = "%s.avi" % save_video_path
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    videoWriter = cv2.VideoWriter(tmp_avi_video_path, fourcc, fps, (width, height))

    for img_path, pose_info in tqdm(zip(img_paths, pose_2d_results)):
        image = cv2.imread(img_path)

        if pose_info["has_person"]:
            joints = pose_info["keypoints"]["pose_keypoints_2d"]
            visual = draw_skeleton(image, joints, radius=6, transpose=False)
            videoWriter.write(visual)

    videoWriter.release()

    convert_avi_to_mp4(tmp_avi_video_path, save_video_path)

