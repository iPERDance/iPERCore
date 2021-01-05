# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import cv2
import numpy as np
import torchvision
from typing import Union, List

from scipy.spatial.transform import Rotation as R


def compute_scaled_size(origin_size, control_size):
    """

    Args:
        origin_size (tuple or List): (h, w) or [h, w]
        control_size (int or float): the final size of the min(h, w)

    Returns:
        scaled_size (tuple or List): (h', w')

    """
    scale_rate = np.sqrt(control_size * control_size / (origin_size[0] * origin_size[1]))
    scaled_size = (int(origin_size[0] * scale_rate), int(origin_size[1] * scale_rate))

    return scaled_size


def read_cv2_img(path):
    """
        Read color images
    Args:
        path (str): Path to image

    Returns:
        img (np.ndarray): color images with RGB channel, and its shape is (H, W, 3).
    """

    img = cv2.imread(path, -1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def load_images(images_paths: Union[str, List[str]], image_size):
    """

    Args:
        images_paths (Union[str, List[str]]):

    Returns:
        images (np.ndarray): shape is (ns, 3, H, W), channel is RGB, and color space is [-1, 1].
    """

    if isinstance(images_paths, str):
        images_paths = [images_paths]

    images = []
    for image_path in images_paths:
        image = read_cv2_img(image_path)
        image = normalize_img(image, image_size=image_size, transpose=True)

        images.append(image)

    images = np.stack(images, axis=0)   # (ns, 3, H, W)
    return images


def read_mask(path, image_size):
    """
        Read mask
    Args:
        path (str): Path to mask

    Returns:
        mask (np.ndarray): mask image with grayscale, and its shape is (1, H, W) in the range of [0, 1]
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (image_size, image_size))
    mask = mask.astype(np.float32) / 255
    mask = np.expand_dims(mask, 0)

    return mask


def load_parse(parse_path, image_size):
    mask = cv2.imread(parse_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (image_size, image_size))
    mask = mask.astype(np.float32) / 255
    mask = np.expand_dims(mask, 0)
    return mask


def load_img_parse(img_path, parse_path, image_size):
    image = transform_img(read_cv2_img(img_path), transpose=True)
    mask = load_parse(parse_path, image_size)
    return image, mask


def save_cv2_img(img, path, image_size=None, normalize=False, transpose=True):

    if transpose:
        img = np.transpose(img, (1, 2, 0))

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))

    if normalize:
        img = (img + 1) / 2.0 * 255
        img = img.astype(np.uint8)

    cv2.imwrite(path, img)
    return img


def transform_img(image, image_size=None, transpose=False):
    if image_size is not None and image_size != image.shape[0]:
        image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32)
    image /= 255.0

    if transpose:
        image = image.transpose((2, 0, 1))

    return image


def normalize_img(image, image_size=None, transpose=False):
    image = transform_img(image, image_size, transpose)
    image *= 2
    image -= 1
    return image


def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def tensor2im(img, imtype=np.uint8, unnormalize=True, idx=0, nrows=None):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(np.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        img += 1.0
        img /= 2.0

    image_numpy = img.numpy()
    # image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy *= 255.0

    return image_numpy.astype(imtype)


def kp_to_bbox_param(kp, vis_thresh=0, diag_len=150.0):
    """
    Finds the bounding box parameters from the 2D keypoints.

    Args:
        kp (Kx3): 2D Keypoints.
        vis_thresh (float): Threshold for visibility.
        diag_len(float): diagonal length of bbox of each person

    Returns:
        [center_x, center_y, scale]
    """
    if kp is None:
        return

    if kp.shape[1] == 3:
        vis = kp[:, 2] > vis_thresh
        if not np.any(vis):
            return
        min_pt = np.min(kp[vis, :2], axis=0)
        max_pt = np.max(kp[vis, :2], axis=0)
    else:
        min_pt = np.min(kp, axis=0)
        max_pt = np.max(kp, axis=0)

    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height < 0.5:
        return
    center = (min_pt + max_pt) / 2.
    scale = diag_len / person_height

    return np.append(center, scale)


def process_hmr_img(im_path, bbox_param, rescale=None, image=None, image_size=256, proc=False):
    """
    Args:
        im_path (str): the path of image.
        image (np.ndarray or None): if it is None, then loading the im_path, else use image.
        bbox_param (3,) : [cx, cy, scale].
        rescale (float, np.ndarray or None): rescale factor.
        proc (bool): the flag to return processed image or not.
        image_size (int):

    Returns:
        proc_img (np.ndarray): if proc is True, return the process image, else return the original image.
    """
    if image is None:
        image = cv2.imread(im_path)

    orig_h, orig_w = image.shape[0:2]
    center = bbox_param[:2]
    scale = bbox_param[2]

    if rescale is not None:
        scale = rescale

    if proc:
        image_scaled, scale_factors = resize_img(image, scale)
        resized_h, resized_w = image_scaled.shape[:2]
    else:
        scale_factors = [scale, scale]
        resized_h = orig_h * scale
        resized_w = orig_w * scale

    center_scaled = np.round(center * scale_factors).astype(np.int)

    if proc:
        # Make sure there is enough space to crop image_size x image_size.
        image_padded = np.pad(
            array=image_scaled,
            pad_width=((image_size,), (image_size,), (0,)),
            mode='edge'
        )
        padded_h, padded_w = image_padded.shape[0:2]
    else:
        padded_h = resized_h + image_size * 2
        padded_w = resized_w + image_size * 2

    center_scaled += image_size

    # Crop image_size x image_size around the center.
    margin = image_size // 2
    start_pt = (center_scaled - margin).astype(int)
    end_pt = (center_scaled + margin).astype(int)
    end_pt[0] = min(end_pt[0], padded_w)
    end_pt[1] = min(end_pt[1], padded_h)

    if proc:
        proc_img = image_padded[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
        # height, width = image_scaled.shape[:2]
    else:
        # height, width = end_pt[1] - start_pt[1], end_pt[0] - start_pt[0]
        proc_img = cv2.resize(image, (image_size, image_size))
        # proc_img = None

    center_scaled -= start_pt

    return {
        # return original too with info.
        'image': proc_img,
        'im_path': im_path,
        'im_shape': (orig_h, orig_w),
        'center': center_scaled,
        'scale': scale,
        'start_pt': start_pt,
    }


def cam_denormalize(cam, N):
    # This is camera in crop image coord.
    new_cam = np.hstack([N * cam[0] * 0.5, cam[1:] + (2. / cam[0]) * 0.5])
    return new_cam


def cam_init2orig(cam, scale, start_pt, N=224):
    """
    Args:
        cam (3,): (s, tx, ty)
        scale (float): scale = resize_h / orig_h
        start_pt (2,): (lt_x, lt_y)
        N (int): hmr_image_size (224) or IMG_SIZE

    Returns:
        cam_orig (3,): (s, tx, ty), camera in original image coordinates.

    """
    # This is camera in crop image coord.
    cam_crop = np.hstack([N * cam[0] * 0.5, cam[1:] + (2. / cam[0]) * 0.5])

    # This is camera in orig image coord
    cam_orig = np.hstack([
        cam_crop[0] / scale,
        cam_crop[1:] + (start_pt - N) / cam_crop[0]
    ])
    # print('cam crop', cam_crop)
    # print('cam orig', cam_orig)
    return cam_orig


def cam_orig2crop_center(cam, scale, start_pt, N=256, normalize=True):
    """
    Args:
        cam (3,): (s, tx, ty), camera in orginal image coordinates.
        scale (float): scale = resize_h / orig_h or (resize_w / orig_w)
        start_pt (2,): (lt_x, lt_y)
        N (int): hmr_image_size (224) or IMG_SIZE
        normalize (bool)

    Returns:

    """
    cam_recrop = np.hstack([
        cam[0] * scale,
        cam[1:] + (N - start_pt) / (scale * cam[0])
    ])
    # print('cam re-crop', cam_recrop)
    if normalize:
        cam_norm = np.hstack([
            cam_recrop[0] * (2. / N),
            cam_recrop[1:] - N / (2 * cam_recrop[0])
        ])
        # print('cam norm', cam_norml)
    else:
        cam_norm = cam_recrop
    return cam_norm


def cam_orig2boxcrop(cam, scale, start_pt, N=256, normalize=True):
    """
    Args:
        cam (3,): (s, tx, ty), camera in orginal image coordinates.
        scale (float): scale = resize_h / orig_h or (resize_w / orig_w)
        start_pt (2,): (lt_x, lt_y)
        N (int): hmr_image_size (224) or IMG_SIZE
        normalize (bool)

    Returns:

    """
    cam_recrop = np.hstack([
        cam[0] * scale,
        cam[1:] - start_pt / cam[0]
    ])
    # print('cam re-crop', cam_recrop)
    if normalize:
        cam_norm = np.hstack([
            cam_recrop[0] * 2. / N,
            cam_recrop[1:] - N / (2 * cam_recrop[0])
        ])
        # print('cam norm', cam_norm)
    else:
        cam_norm = cam_recrop
    return cam_norm


def cam_process(cam_init, scale_150, start_pt_150, scale_proc, start_pt_proc, HMR_IMG_SIZE=224, IMG_SIZE=256):
    """
    Args:
        cam_init:
        scale_150:
        start_pt_150:
        scale_proc:
        start_pt_proc:
        HMR_IMG_SIZE:
        IMG_SIZE:

    Returns:

    """
    # print(HMR_IMG_SIZE, IMG_SIZE)
    cam_orig = cam_init2orig(cam_init, scale=scale_150, start_pt=start_pt_150, N=HMR_IMG_SIZE)
    cam_crop = cam_orig2crop_center(cam_orig, scale=scale_proc, start_pt=start_pt_proc, N=IMG_SIZE, normalize=True)
    return cam_orig, cam_crop


def intrinsic_mtx(f, c):
    """
    Obtain intrisic camera matrix.
    Args:
        f: np.array, 1 x 2, the focus lenth of camera, (fx, fy)
        c: np.array, 1 x 2, the center of camera, (px, py)
    Returns:
        - cam_mat: np.array, 3 x 3, the intrisic camera matrix.
    """
    return np.array([[f[1], 0, c[1]],
                     [0, f[0], c[0]],
                     [0, 0, 1]], dtype=np.float32)


def extrinsic_mtx(rt, t):
    """
    Obtain extrinsic matrix of camera.
    Args:
        rt: np.array, 1 x 3, the angle of rotations.
        t: np.array, 1 x 3, the translation of camera center.
    Returns:
        - ext_mat: np.array, 3 x 4, the extrinsic matrix of camera.
    """
    # R is (3, 3)
    R = cv2.Rodrigues(rt)[0]
    t = np.reshape(t, newshape=(3, 1))
    Rc = np.dot(R, t)
    ext_mat = np.hstack((R, -Rc))
    ext_mat = np.vstack((ext_mat, [0, 0, 0, 1]))
    ext_mat = ext_mat.astype(np.float32)
    return ext_mat


def extrinsic(rt, t):
    """
    Obtain extrinsic matrix of camera.
    Args:
        rt: np.array, 1 x 3, the angle of rotations.
        t: np.array, 1 x 3, or (3,) the translation of camera center.
    Returns:
        - R: np.ndarray, 3 x 3
        - t: np.ndarray, 1 x 3
    """
    R = cv2.Rodrigues(rt)[0]
    t = np.reshape(t, newshape=(1, 3))
    return R, t


def euler2matrix(rt):
    """
    Obtain rotation matrix from euler angles
    Args:
        rt: np.array, (3,)
    Returns:
        R: np.array, (3,3)
    """
    Rx = np.array([[1, 0,             0],
                   [0, np.cos(rt[0]), -np.sin(rt[0])],
                   [0, np.sin(rt[0]), np.cos(rt[0])]], dtype=np.float32)

    Ry = np.array([[np.cos(rt[1]),     0,       np.sin(rt[1])],
                   [0,                 1,       0],
                   [-np.sin(rt[1]),    0,       np.cos(rt[1])]], dtype=np.float32)

    Rz = np.array([[np.cos(rt[2]),     -np.sin(rt[2]),       0],
                   [np.sin(rt[2]),      np.cos(rt[2]),       0],
                   [0,                              0,       1]], dtype=np.float32)

    return np.dot(Rz, np.dot(Ry, Rx))


def get_rotated_smpl_pose(pose, theta):
    """
    :param pose: (72,)
    :param theta: rotation angle of y axis
    :return:
    """
    global_pose = pose[:3]
    R, _ = cv2.Rodrigues(global_pose)
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    new_R = np.matmul(R, Ry)
    new_global_pose, _ = cv2.Rodrigues(new_R)
    new_global_pose = new_global_pose.reshape(3)

    rotated_pose = pose.copy()
    rotated_pose[:3] = new_global_pose

    return rotated_pose


def rotvec2mat(rotvec):
    """
    Args:
        quat (np.ndarray): (N, 72)

    Returns:
        rot_vec (np.ndarray): (N, 24, 3, 3)
    """
    r = R.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
    r = r.reshape(-1, 24, 3, 3)

    return r


def quat2rotvec(quat):
    """
    Args:
        quat (np.ndarray): (N, 96)

    Returns:
        rot_vec (np.ndarray): (N, 72)
    """

    rotvec_pos = R.from_quat(quat.reshape(-1, 4)).as_rotvec()
    rotvec_pos = rotvec_pos.reshape(-1, 24 * 3)

    return rotvec_pos


def rotvec2quat(rotvec):
    """

    Args:
        rotvec (np.ndarray): (N, 72)

    Returns:
        qaut (np.ndarray): (N, 96)
    """

    n = rotvec.shape[0]
    quat_pos = R.from_rotvec(rotvec.reshape(-1, 3)).as_quat()
    quat_pos = quat_pos.reshape(n, -1)

    return quat_pos


def rotvec2euler(rotvec):
    """

    Args:
        rotvec (np.ndarray): (N, 72)

    Returns:
        euler (np.ndarray): (N, 72)
    """
    n = rotvec.shape[0]
    r = R.from_rotvec(rotvec.reshape(-1, 3)).as_euler("xyz", degrees=False)
    r = r.reshape(n, -1)

    return r


def euler2rotvec(euler):
    """

    Args:
        euler (np.ndarray): (N, 72)

    Returns:
        euler (np.ndarray): (N, 72)
    """
    n = euler.shape[0]
    r = R.from_euler("xyz", euler.reshape(-1, 3), degrees=False)
    r = r.as_rotvec()
    r = r.reshape(n, -1)

    return r


if __name__ == '__main__':
    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    R = euler2matrix(np.array([0, 90, 0], dtype=np.float32))

    print(isRotationMatrix(R))

