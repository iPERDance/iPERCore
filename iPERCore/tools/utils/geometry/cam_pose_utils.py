# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import numpy as np
import torch
from typing import Union


class WeakPerspectiveCamera(object):

    def __init__(self, smpl):
        self.smpl = smpl
        self.infer_smpl_batch_size = 50
        self.jump_up_threshold = 0.2
        self.jump_down_threshold = 0.1

    @staticmethod
    def cam_swap(src_cam, ref_cam, first_cam=None, strategy="smooth"):
        """
        Swap the camera between the source and the reference poses, according to the different swapping
        strategies, including, `smooth`, `source`, `ref_txty`, and `copy`

        Args:
            src_cam (torch.tensor): (bs, 3);
            ref_cam (torch.tesnor): (bs, 3);
            first_cam (torch.tensor or None): if strategy is `smooth`, we must provide with first_cam (bs, 3);
            strategy (str): swapping strategies, including `smooth`, `source`, `ref_txty`, and `copy`.

        Returns:
            cam (torch.tensor): (bs, 3)
        """

        if strategy == "smooth":
            cam = src_cam.clone()
            delta_xy = ref_cam[:, 1:] - first_cam[:, 1:]
            cam[:, 1:] += delta_xy

            # scale
            cam[:, 0] = cam[:, 0] * ref_cam[:, 0] / first_cam[:, 0]

        elif strategy == "ref_txty":
            cam = src_cam.clone()
            cam[:, 1:] = ref_cam[:, 1:]

        elif strategy == "source":
            cam = src_cam

        else:
            cam = ref_cam

        return cam

    def stabilize(self, smpls):
        """

        Args:
            smpls (torch.tensor): (bs, 85)

        Returns:
            stable_smpls (torch.tensor): (bs, 85)
        """

        cam = smpls[:, 0:3]
        pose = smpls[:, 3:-10]
        shape = smpls[:, -10:]

        new_cam = torch.zeros_like(cam, device=smpls.device)
        new_cam[:, 0] = 1
        new_cam[:, 1] = 0

        cam_y = cam[:, 2]
        ground_y = cam_y[0]

        shape = shape[0:1, :].repeat(pose.shape[0], 1)

        """
        infer foot y
        """
        # (bs, )
        foot_y = self.infer_smpl_foot_y(pose, shape)
        # (bs, )
        origin_final_foot_y = foot_y + cam_y
        # numpy (bs, )
        origin_final_foot_y = origin_final_foot_y.detach().cpu().numpy()
        jump_info_list, jump_mask = self.get_jump_mask(origin_final_foot_y)

        """
        denoise
        """
        denoise = -foot_y + foot_y[0]

        new_cam_y = ground_y + denoise
        for start_idx, end_idx in jump_info_list:
            jump_part = cam_y[start_idx:end_idx + 1].clone()
            new_cam_y[start_idx:end_idx + 1] = torch.min(jump_part, new_cam_y[start_idx:end_idx + 1])

        new_cam[:, 2] = new_cam_y

        stable_smpls = torch.cat([new_cam, pose, shape], dim=1)
        return stable_smpls

    @torch.no_grad()
    def infer_smpl_foot_y(self, pose, shape):
        """

        Args:
            pose (torch.tensor): (bs, 72)
            shape (torch.tensor): (bs, 10)

        Returns:
            foot_y (torch.tensor): (bs,)

        """

        N = pose.shape[0]
        batch_size = self.infer_smpl_batch_size
        foot_y_list = []

        for i in range(int(np.ceil(N / batch_size))):
            batch_pose = pose[i * batch_size:(i + 1) * batch_size].contiguous()
            batch_shape = shape[i * batch_size:(i + 1) * batch_size].contiguous()

            batch_verts, batch_joints, _ = self.smpl(batch_shape, batch_pose, get_skin=True)

            # (batch_size, )
            batch_foot_y = batch_verts[:, :, 1].max(dim=1, keepdim=True)[0][:, 0]
            foot_y_list.append(batch_foot_y)

        foot_y = torch.cat(foot_y_list, dim=0)
        return foot_y

    @staticmethod
    def get_checkpoints(y):
        """

        Args:
            y (torch.Tensor or np.ndarray): (bs, )

        Returns:
            checkpoints (list): all checkpoints of the jumpint points.

        """

        y_len = len(y)
        checkpoints = [0]

        for i in range(1, y_len - 1):
            pre = y[i] - y[i - 1]
            cur = y[i + 1] - y[i]

            if pre * cur < 0:
                checkpoints.append(i)
        checkpoints.append(y_len - 1)

        return checkpoints

    def get_jump_mask(self, final_foot_y):
        """

        Args:
            final_foot_y (np.ndarray): (bs,)

        Returns:
            jump_info_list (list of tuple): [(start_idx, end_idx), ...];
            jump_mask (np.ndarray): (bs,), 1 for jumping status.
        """

        frame_num = final_foot_y.shape[0]
        jump_info_list = []
        ground_y = final_foot_y[0]

        checkpoints = self.get_checkpoints(final_foot_y)
        jump_flag = False
        start_idx, end_idx = None, None

        for ckpt_idx in range(1, len(checkpoints)):
            ckpt_i = checkpoints[ckpt_idx]
            ckpt_i_1 = checkpoints[ckpt_idx - 1]

            y_i = final_foot_y[ckpt_i]
            y_i_1 = final_foot_y[ckpt_i_1]

            if y_i - y_i_1 < 0 and abs(y_i - y_i_1) > self.jump_up_threshold:
                jump_flag = True
                # if start_jumping frame y > ground_y, that is noise, ignore it
                start_idx = None
                for frame_idx in range(ckpt_i_1, ckpt_i):
                    frame_y = final_foot_y[frame_idx]
                    if frame_y < ground_y:
                        start_idx = frame_idx
                        break
                if start_idx == None:
                    start_idx = ckpt_i_1

            elif jump_flag:
                dis = abs(y_i - final_foot_y[start_idx])
                if y_i < final_foot_y[start_idx] and dis > self.jump_down_threshold:
                    continue

                jump_flag = False
                end_idx = ckpt_i
                jump_info_list.append((start_idx, end_idx))
                start_idx, end_idx = None, None

        # spacial case: video has finished but person is still jumping
        if jump_flag:
            end_idx = frame_num - 1
            jump_info_list.append((start_idx, end_idx))

        jump_mask = np.zeros((frame_num,))
        for start_idx, end_idx in jump_info_list:
            jump_mask[start_idx:end_idx + 1] = 1

        return jump_info_list, jump_mask


def cam_init2orig(cam, scale: Union[float, torch.Tensor], start_pt: torch.Tensor, N=224):
    """
    Args:
        cam (bs, 3): (s, tx, ty)
        scale (bs,): scale = resize_h / orig_h
        start_pt (bs, 2): (lt_x, lt_y)
        N (int): hmr_image_size (224) or IMG_SIZE

    Returns:
        cam_orig (bs, 3): (s, tx, ty), camera in original image coordinates.

    """

    # This is camera in crop image coord.
    cam_crop = torch.cat(
        [N * cam[:, 0:1] * 0.5, cam[:, 1:] + (2. / cam[:, 0:1]) * 0.5],
        dim=1
    )

    # This is camera in orig image coord
    cam_orig = torch.cat(
        [cam_crop[:, 0:1] / scale, cam_crop[:, 1:] + (start_pt - N) / cam_crop[:, 0:1]],
        dim=1
    )

    return cam_orig


def cam_norm(cam, N):
    cam_norm = torch.cat([
        cam[:, 0:1] * (2. / N),
        cam[:, 1:] - N / (2 * cam[:, 0:1])
    ], dim=1)
    return cam_norm

