# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import cv2
import glob
import shutil
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import numpy as np
import subprocess
import json

default_ffmpeg_exe_path = "ffmpeg"
default_ffprobe_exe_path = "ffprobe"
default_ffmpeg_vcodec = "h264"
default_ffmpeg_pix_fmt = "yuv420p"


def auto_unzip_fun(x, f):
    return f(*x)


def convert_avi_to_mp4(tmp_avi_video_path, output_mp4_path):
    """

    Args:
        tmp_avi_video_path:
        output_mp4_path:

    Returns:

    """

    global default_ffmpeg_vcodec, default_ffmpeg_pix_fmt, default_ffmpeg_exe_path

    ffmpeg_exc_path = os.environ.get("ffmpeg_exe_path", default_ffmpeg_exe_path)
    vcodec = os.environ.get("ffmpeg_vcodec", default_ffmpeg_vcodec)

    cmd = [
        ffmpeg_exc_path,
        "-y",
        "-i", tmp_avi_video_path,
        "-vcodec", vcodec,
        output_mp4_path,
        "-loglevel", "quiet"
    ]
    print(" ".join(cmd))
    subprocess.call(cmd)
    os.remove(tmp_avi_video_path)


def make_video(output_mp4_path, img_path_list, save_frames_dir=None, fps=24, pool_size=40):
    """
    output_path is the final mp4 name
    img_dir is where the images to make into video are saved.
    """

    global default_ffmpeg_vcodec, default_ffmpeg_pix_fmt, default_ffmpeg_exe_path

    ffmpeg_exc_path = os.environ.get("ffmpeg_exe_path", default_ffmpeg_exe_path)
    vcodec = os.environ.get("ffmpeg_vcodec", default_ffmpeg_vcodec)

    first_img = cv2.imread(img_path_list[0])
    h, w = first_img.shape[:2]

    tmp_avi_video_path = "%s.avi" % output_mp4_path
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    videoWriter = cv2.VideoWriter(tmp_avi_video_path, fourcc, fps, (w, h))
    args_list = [(img_path,) for img_path in img_path_list]
    pool_size = min(pool_size, os.cpu_count())
    with Pool(pool_size) as p:
        for img in tqdm(p.imap(partial(auto_unzip_fun, f=cv2.imread), args_list), total=len(args_list)):
            videoWriter.write(img)
    videoWriter.release()

    if save_frames_dir:
        for i, img_path in enumerate(img_path_list):
            shutil.copy(img_path, "%s/%.8d.jpg" % (save_frames_dir, i))

    cmd = [
        ffmpeg_exc_path,
        "-y",
        "-i", tmp_avi_video_path,
        "-vcodec", vcodec,
        output_mp4_path,
        "-loglevel", "quiet"
    ]
    print(" ".join(cmd))
    subprocess.call(cmd)
    os.remove(tmp_avi_video_path)


def fuse_image(img_path_list, row_num, col_num):
    assert len(img_path_list) == row_num * col_num

    img_list = [cv2.imread(img_path) for img_path in img_path_list]

    row_imgs = []
    for i in range(row_num):
        col_imgs = img_list[i * col_num: (i + 1) * col_num]
        col_img = np.concatenate(col_imgs, axis=1)
        row_imgs.append(col_img)

    fused_img = np.concatenate(row_imgs, axis=0)
    return fused_img


def fuse_video(video_frames_path_list, output_mp4_path, row_num, col_num, fps=24, pool_size=40):
    global default_ffmpeg_vcodec, default_ffmpeg_pix_fmt, default_ffmpeg_exe_path

    ffmpeg_exc_path = os.environ.get("ffmpeg_exe_path", default_ffmpeg_exe_path)
    vcodec = os.environ.get("ffmpeg_vcodec", default_ffmpeg_vcodec)

    assert len(video_frames_path_list) == row_num * col_num

    frame_num = len(video_frames_path_list[0])
    first_img = cv2.imread(video_frames_path_list[0][0])
    h, w = first_img.shape[:2]
    fused_h, fused_w = h * row_num, w * col_num

    args_list = []
    for frame_idx in range(frame_num):
        fused_frame_path_list = [video_frames[frame_idx] for video_frames in video_frames_path_list]
        args_list.append((fused_frame_path_list, row_num, col_num))

    tmp_avi_video_path = "%s.avi" % output_mp4_path
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    videoWriter = cv2.VideoWriter(tmp_avi_video_path, fourcc, fps, (fused_w, fused_h))
    pool_size = min(pool_size, os.cpu_count())
    with Pool(pool_size) as p:
        for img in tqdm(p.imap(partial(auto_unzip_fun, f=fuse_image), args_list), total=len(args_list)):
            videoWriter.write(img)
    videoWriter.release()

    # os.system("ffmpeg -y -i %s -vcodec h264 %s > /dev/null 2>&1" % (tmp_avi_video_path, output_mp4_path))
    # os.system("rm %s" % (tmp_avi_video_path))

    cmd = [
        ffmpeg_exc_path,
        "-y",
        "-i", tmp_avi_video_path,
        "-vcodec", vcodec,
        output_mp4_path
    ]
    print(" ".join(cmd))
    subprocess.call(cmd)
    os.remove(tmp_avi_video_path)


def merge(src_img, ref_img_path, out_img_path, pad):
    h, w = src_img.shape[:2]
    image_size = h

    ref_img = cv2.imread(ref_img_path)
    out_img = cv2.imread(out_img_path)

    if ref_img.shape[0] != image_size and ref_img.shape[1] != image_size:
        ref_img = cv2.resize(ref_img, (image_size, image_size))

    if out_img.shape[0] != image_size and out_img.shape[1] != image_size:
        out_img = cv2.resize(out_img, (image_size, image_size))

    # print(src_img.shape, ref_img.shape, out_img.shape)
    merge_img = np.concatenate([src_img, pad, ref_img, pad, out_img], axis=1)

    return merge_img


def merge_multi_outs(src_img, ref_img_path, multi_out_paths, pad):
    h, w = src_img.shape[:2]
    image_size = h

    merge_img = []
    merge_img.append(src_img)

    ref_img = cv2.imread(ref_img_path)
    if ref_img.shape[0] != image_size and ref_img.shape[1] != image_size:
        ref_img = cv2.resize(ref_img, (image_size, image_size))

    merge_img.append(pad)
    merge_img.append(ref_img)

    for out_img_path in multi_out_paths:
        out_img = cv2.imread(out_img_path)
        if out_img.shape[0] != image_size and out_img.shape[1] != image_size:
            out_img = cv2.resize(out_img, (image_size, image_size))

        merge_img.append(pad)
        merge_img.append(out_img)

    # print(src_img.shape, ref_img.shape, out_img.shape)
    merge_img = np.concatenate(merge_img, axis=1)

    return merge_img


def merge_src_out(src_img, out_img_path, pad):
    h, w = src_img.shape[:2]
    image_size = h

    out_img = cv2.imread(out_img_path)

    if out_img.shape[0] != image_size and out_img.shape[1] != image_size:
        out_img = cv2.resize(out_img, (image_size, image_size))

    # print(src_img.shape, ref_img.shape, out_img.shape)
    merge_img = np.concatenate([src_img, pad, out_img], axis=1)

    return merge_img


def load_image(image_path, image_size=512):
    """

    Args:
        image_path (str):
        image_size (int):

    Returns:
        image (np.ndarray): (image_size, image_size, 3), BGR channel space, in the range of [0, 255], np.uint8.
    """

    if image_path:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size, image_size))
    else:
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    return image


def fuse_one_image(img_paths, image_size):
    return load_image(img_paths[0], image_size)


def fuse_two_images(img_paths, image_size):
    """

    Args:
        img_paths (list of str):
        image_size (int):

    Returns:
        fuse_img (np.ndarray): (image_size // 2, image_size, 3), BGR channel space, in the range of [0, 255], np.uint8.
    """

    img_size = image_size // 2

    img_1 = load_image(img_paths[0], img_size)
    img_2 = load_image(img_paths[1], img_size)

    fuse_img = np.concatenate([img_1, img_2], axis=0)

    return fuse_img


def fuse_four_images(img_paths, image_size):
    """

    Args:
        img_paths (list of str):
        image_size (int):

    Returns:
        fuse_img (np.ndarray): (image_size, image_size, 3), BGR channel space, in the range of [0, 255], np.uint8.
    """

    fuse_img_1 = fuse_two_images(img_paths[0:2], image_size)
    fuse_img_2 = fuse_two_images(img_paths[2:4], image_size)

    fuse_img = np.concatenate([fuse_img_1, fuse_img_2], axis=1)
    return fuse_img


def fuse_eight_images(img_paths, image_size):
    """

    Args:
        img_paths (List[Union[str, None]]):
        image_size (int):

    Returns:
        fuse_img (np.ndarray): (image_size // 2, image_size, 3), BGR channel space, in the range of [0, 255], np.uint8.
    """

    fuse_img_1 = fuse_two_images(img_paths[0:4], image_size // 2)
    fuse_img_2 = fuse_two_images(img_paths[4:8], image_size // 2)

    fuse_img = np.concatenate([fuse_img_1, fuse_img_2], axis=0)
    return fuse_img


def fuse_source(all_src_img_paths, image_size=512):
    """

    Args:
        all_src_img_paths (list of str): the list of source image paths. Denotes ns as the number of source images,
            if ns == 1, we will display one single image;
            if ns == 2, we will display two images;
            if ns == 3, we will pad them to four images and display them;
            if ns == 4, we will display four images;
            if ns == 5, we will only take the first four images and display them;
            if ns == 6, we will only take the first four images and display them;
            if ns == 7, we will pad them to eight images and display them;
            if ns >= 8, we will display eight images.

        image_size (int): the final image resolution, (image_size, image_size, 3)

    Returns:
        fuse_img (np.ndarray): (image_size, image_size, 3), BGR channel space, in the range of [0, 255], np.uint8.
    """

    ns = len(all_src_img_paths)

    if ns == 1:
        fuse_img = load_image(all_src_img_paths[0], image_size)

    elif ns == 2:
        fuse_img = fuse_two_images(all_src_img_paths, image_size)

    elif ns == 3:
        # pad it to 4 images
        visual_src_img_paths = all_src_img_paths + [""]
        fuse_img = fuse_four_images(visual_src_img_paths, image_size)

    elif ns == 4:
        fuse_img = fuse_four_images(all_src_img_paths, image_size)

    elif ns == 5:
        # take only 4 images
        visual_src_img_paths = all_src_img_paths[0:4]
        fuse_img = fuse_four_images(visual_src_img_paths, image_size)

    elif ns == 6:
        # take only 4 images
        visual_src_img_paths = all_src_img_paths[0:4]
        fuse_img = fuse_four_images(visual_src_img_paths, image_size)

    elif ns == 7:
        # pad it to 8 images
        visual_src_img_paths = all_src_img_paths + [""]
        fuse_img = fuse_four_images(visual_src_img_paths, image_size)

    elif ns == 8:
        fuse_img = fuse_eight_images(all_src_img_paths, image_size)

    elif ns > 8:
        fuse_img = fuse_eight_images(all_src_img_paths[0:8], image_size)

    else:
        raise ValueError(f"the number of source images must > 0, while the current value is {ns}.")

    return fuse_img


def fuse_source_output(output_mp4_path, src_img_paths, out_img_paths,
                       audio_path=None, image_size=512, pad=10, fps=25, pool_size=15):
    global default_ffmpeg_vcodec, default_ffmpeg_pix_fmt, default_ffmpeg_exe_path

    ffmpeg_exc_path = os.environ.get("ffmpeg_exe_path", default_ffmpeg_exe_path)
    vcodec = os.environ.get("ffmpeg_vcodec", default_ffmpeg_vcodec)

    fused_src_img = fuse_source(src_img_paths, image_size)
    pad_region = np.zeros((image_size, pad, 3), dtype=np.uint8)

    pool_size = min(pool_size, os.cpu_count())
    tmp_avi_video_path = "%s.avi" % output_mp4_path
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    W = fused_src_img.shape[1] + image_size + pad
    videoWriter = cv2.VideoWriter(tmp_avi_video_path, fourcc, fps, (W, image_size))

    total = len(out_img_paths)
    with ProcessPoolExecutor(pool_size) as pool:
        for img in tqdm(pool.map(merge_src_out, [fused_src_img] * total, out_img_paths, [pad_region] * total)):
            videoWriter.write(img)

    videoWriter.release()

    if audio_path is not None and audio_path and os.path.exists(audio_path):
        fuse_video_audio_output(tmp_avi_video_path, audio_path, output_mp4_path)
        os.remove(tmp_avi_video_path)
    else:
        # os.system("ffmpeg -y -i %s -vcodec h264 %s > /dev/null 2>&1" % (tmp_avi_video_path, output_mp4_path))
        # os.system("rm %s" % tmp_avi_video_path)

        cmd = [
            ffmpeg_exc_path,
            "-y",
            "-i", tmp_avi_video_path,
            "-vcodec", vcodec,
            output_mp4_path,
            "-loglevel", "quiet"
        ]
        print(" ".join(cmd))
        subprocess.call(cmd)
        os.remove(tmp_avi_video_path)


def fuse_source_reference_output(output_mp4_path, src_img_paths, ref_img_paths, out_img_paths,
                                 audio_path=None, image_size=512, pad=10, fps=25, pool_size=15):
    global default_ffmpeg_vcodec, default_ffmpeg_pix_fmt, default_ffmpeg_exe_path

    ffmpeg_exc_path = os.environ.get("ffmpeg_exe_path", default_ffmpeg_exe_path)
    vcodec = os.environ.get("ffmpeg_vcodec", default_ffmpeg_vcodec)

    total = len(ref_img_paths)
    assert total == len(out_img_paths), "{} != {}".format(total, len(out_img_paths))

    fused_src_img = fuse_source(src_img_paths, image_size)
    pad_region = np.zeros((image_size, pad, 3), dtype=np.uint8)

    pool_size = min(pool_size, os.cpu_count())
    tmp_avi_video_path = "%s.avi" % output_mp4_path
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    W = fused_src_img.shape[1] + (image_size + pad) * 2
    videoWriter = cv2.VideoWriter(tmp_avi_video_path, fourcc, fps, (W, image_size))

    with ProcessPoolExecutor(pool_size) as pool:
        for img in tqdm(pool.map(merge, [fused_src_img] * total,
                                 ref_img_paths, out_img_paths, [pad_region] * total)):
            videoWriter.write(img)

    videoWriter.release()

    if audio_path is not None and audio_path and os.path.exists(audio_path):
        fuse_video_audio_output(tmp_avi_video_path, audio_path, output_mp4_path)
        os.remove(tmp_avi_video_path)
    else:
        # os.system("ffmpeg -y -i %s -vcodec h264 %s > /dev/null 2>&1" % (tmp_avi_video_path, output_mp4_path))
        # os.system("rm %s" % tmp_avi_video_path)

        cmd = [
            ffmpeg_exc_path,
            "-y",
            "-i", tmp_avi_video_path,
            "-vcodec", vcodec,
            output_mp4_path,
            "-loglevel", "quiet"
        ]
        print(" ".join(cmd))
        subprocess.call(cmd)
        os.remove(tmp_avi_video_path)


def fuse_src_ref_multi_outputs(output_mp4_path, src_img_paths, ref_img_paths, multi_out_img_paths, audio_path=None,
                               output_dir=None, image_size=512, pad=10, pad_val=0, fps=25, pool_size=15):

    global default_ffmpeg_vcodec, default_ffmpeg_pix_fmt, default_ffmpeg_exe_path

    ffmpeg_exc_path = os.environ.get("ffmpeg_exe_path", default_ffmpeg_exe_path)
    vcodec = os.environ.get("ffmpeg_vcodec", default_ffmpeg_vcodec)

    total = len(ref_img_paths)

    assert total == len(multi_out_img_paths), "{} != {}".format(total, len(multi_out_img_paths))

    num_outs = len(multi_out_img_paths[0])

    fused_src_img = fuse_source(src_img_paths, image_size)
    pad_region = np.zeros((image_size, pad, 3), dtype=np.uint8) + pad_val

    pool_size = min(pool_size, os.cpu_count())
    tmp_avi_video_path = "%s.avi" % output_mp4_path
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    W = fused_src_img.shape[1] + (image_size + pad) * (num_outs + 1)
    videoWriter = cv2.VideoWriter(tmp_avi_video_path, fourcc, fps, (W, image_size))

    i = 0
    with ProcessPoolExecutor(pool_size) as pool:
        for img in tqdm(pool.map(merge_multi_outs, [fused_src_img] * total,
                                 ref_img_paths, multi_out_img_paths, [pad_region] * total)):

            videoWriter.write(img)

            if output_dir is not None:
                cv2.imwrite(os.path.join(output_dir, "{:0>8}.png".format(i)), img)
                i += 1

    videoWriter.release()

    if audio_path is not None and audio_path and os.path.exists(audio_path):
        fuse_video_audio_output(tmp_avi_video_path, audio_path, output_mp4_path)
        os.remove(tmp_avi_video_path)
    else:
        # os.system("ffmpeg -y -i %s -vcodec h264 %s > /dev/null 2>&1" % (tmp_avi_video_path, output_mp4_path))
        # os.system("rm %s" % tmp_avi_video_path)

        cmd = [
            ffmpeg_exc_path,
            "-y",
            "-i", tmp_avi_video_path,
            "-vcodec", vcodec,
            output_mp4_path,
            "-loglevel", "quiet"
        ]
        print(" ".join(cmd))
        subprocess.call(cmd)
        os.remove(tmp_avi_video_path)


def fuse_video_audio_output(video_path, audio_path, out_path):
    global default_ffmpeg_vcodec, default_ffmpeg_pix_fmt, default_ffmpeg_exe_path

    ffmpeg_exc_path = os.environ.get("ffmpeg_exe_path", default_ffmpeg_exe_path)
    vcodec = os.environ.get("ffmpeg_vcodec", default_ffmpeg_vcodec)

    # os.system("ffmpeg -y -i {video_path} -i {audio_path} -vcodec h264 -shortest  -strict -2 {out_path}".format(
    #     video_path=video_path, audio_path=audio_path, out_path=out_path))
    cmd = [
        ffmpeg_exc_path,
        "-y",
        "-i", video_path,
        "-i", audio_path,
        "-vcodec", vcodec,
        "-shortest",
        "-strict", "-2",
        out_path,
        "-loglevel", "quiet"
    ]
    print(" ".join(cmd))
    subprocess.call(cmd)


def video2frames(vid_path, out_dir):
    """
    Extracts all frames from the video at vid_path and saves them inside of
    out_dir.
    """

    global default_ffmpeg_vcodec, default_ffmpeg_pix_fmt, default_ffmpeg_exe_path

    ffmpeg_exc_path = os.environ.get("ffmpeg_exe_path", default_ffmpeg_exe_path)

    imgs = glob.glob(os.path.join(out_dir, "*.png"))
    length = len(imgs)
    if length > 0:
        print("Writing frames to file: done!")
        return out_dir

    print("{} Writing frames to file".format(vid_path))

    cmd = [
        ffmpeg_exc_path,
        "-i", vid_path,
        "-start_number", "0",
        "{temp_dir}/frame_%08d.png".format(temp_dir=out_dir),
    ]
    print(" ".join(cmd))
    subprocess.call(cmd)

    return out_dir


def frames2video(imgs_dir, vid_path, fps=25, prefix="frame", num_digits=8, ffmpeg_exc_path="ffmpeg"):
    global default_ffmpeg_vcodec, default_ffmpeg_pix_fmt, default_ffmpeg_exe_path

    ffmpeg_exc_path = os.environ.get("ffmpeg_exe_path", default_ffmpeg_exe_path)
    vcodec = os.environ.get("ffmpeg_vcodec", default_ffmpeg_vcodec)
    pix_fmt = os.environ.get("ffmpeg_pix_fmt", default_ffmpeg_pix_fmt)

    # ffmpeg -f image2 -i /home/ttwang/images/image%d.jpg tt.mp4
    images_names = os.listdir(imgs_dir)
    image_name = images_names[0]
    suffix = image_name.split(".")[-1]

    cmd = [
        ffmpeg_exc_path,
        "-y",
        "-r", str(fps),
        "-f", "image2",
        "-i", "{temp_dir}/{prefix}_%{num_digits}d.{suffix}".format(
            temp_dir=imgs_dir, prefix=prefix, num_digits=num_digits, suffix=suffix),
        "-vcodec", vcodec,
        "-pix_fmt", pix_fmt,
        vid_path
    ]
    print(" ".join(cmd))
    subprocess.call(cmd)

    return imgs_dir


def extract_audio_from_video(video_path, save_audio_path):
    """

    Extract the audio from a video.

    Args:
        video_path (str): the input video path;
        save_audio_path (str): the saved audio path.

    Returns:

    """

    global default_ffmpeg_exe_path

    ffmpeg_exe_path = os.environ.get("ffmpeg_exe_path", default_ffmpeg_exe_path)

    # os.system(f"ffmpeg -i {video_path}  -ab 160k -ac 2 -ar 44100 -vn {save_audio_path} > /dev/null 2>&1")

    cmd = [
        ffmpeg_exe_path, "-y",
        "-i", video_path,
        "-ab", "160k",
        "-ac", "2",
        "-ar", "44100",
        "-vn", save_audio_path,
        "-loglevel", "quiet"
    ]

    print(" ".join(cmd))
    subprocess.run(cmd)


def get_video_fps(video_path, ret_type="float"):
    """
    Get the fps of the video.

    Args:
        video_path (str): the video path;
        ret_type (str): the return type, it supports `str`, `float`, and `tuple` (numerator, denominator).

    Returns:
        --fps (str): if ret_type is `str`.
        --fps (float): if ret_type is `float`.
        --fps (tuple): if ret_type is tuple, (numerator, denominator).
    """

    global default_ffprobe_exe_path

    ffprobe_exe_path = os.environ.get("ffprobe_exe_path", default_ffprobe_exe_path)

    cmd = [
        ffprobe_exe_path, "-v", "error", "-select_streams", "v", "-of",
        "default=noprint_wrappers=1:nokey=1", "-show_entries", "stream=r_frame_rate",
        video_path
    ]

    print(" ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    fps = result.stdout.decode("utf-8").strip()

    # e.g. 30/1
    numerator, denominator = map(lambda x: int(x), fps.split("/"))
    if ret_type == "float":
        return numerator / denominator
    elif ret_type == "str":
        return str(numerator / denominator)
    else:
        return numerator, denominator


def check_video_has_audio(video_path):
    """
    Check the video has audio or not.

    Args:
        video_path (str): the video path.

    Returns:
        has_audio (bool): True, if it has audio; otherwise it returns False.
    """

    global default_ffprobe_exe_path

    ffprobe_exe_path = os.environ.get("ffprobe_exe_path", default_ffprobe_exe_path)

    # cmd = f"{ffprobe_exe_path} -show_entries stream=codec_type -of json {video_path} -loglevel error"
    cmd = [
        ffprobe_exe_path,
        "-show_entries",
        "stream=codec_type",
        "-of", "json",
        video_path,
        "-loglevel", "error"
    ]
    print(" ".join(cmd))
    results = subprocess.run(cmd, stdout=subprocess.PIPE)

    result_params = json.loads(results.stdout)
    # print(result_params)

    has_audio = False
    for stream_dict in result_params["streams"]:
        if stream_dict["codec_type"] == "audio":
            has_audio = True
            break

    return has_audio

