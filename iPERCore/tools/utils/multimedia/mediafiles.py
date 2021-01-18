# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os


IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG",
    ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

VIDEO_EXTENSION = [
    ".mp4", ".MP4", ".avi", ".AVI", "mkv", "webm"
]


def is_image_file(filename):
    global IMG_EXTENSIONS
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_video_file(filename):
    global VIDEO_EXTENSION
    return os.path.isfile(filename) and any(filename.endswith(extension) for extension in VIDEO_EXTENSION)


