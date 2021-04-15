# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import torch
from torch.optim import lr_scheduler

from iPERCore.models import FlowComposition, BaseModel
from iPERCore.tools.human_digitalizer.bodynets import SMPL
from iPERCore.tools.utils.morphology import morph


class BaseTrainerModel(BaseModel):

    def __init__(self, opt):
        super(BaseTrainerModel, self).__init__(opt)

        self._name = "BaseTrainerModel"

    def set_input(self, *inputs):
        raise NotImplementedError

    def set_train(self):
        raise NotImplementedError

    def set_eval(self):
        raise NotImplementedError

    def optimize_parameters(self):
        raise NotImplementedError

    def get_current_visuals(self):
        raise NotImplementedError

    def get_current_errors(self):
        raise NotImplementedError

    def get_current_scalars(self):
        raise NotImplementedError

    def save(self, label):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = f"opt_iter_{epoch_label}_id_{optimizer_label}.pth"
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, optimizer, optimizer_label, epoch_label, device="cpu"):
        load_filename = f"opt_iter_{epoch_label}_id_{optimizer_label}.pth"
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(load_path), "Weights file not found. %s " \
                                          "Have you trained a model!? We are not providing one" % load_path

        optimizer.load_state_dict(torch.load(load_path, map_location=device))
        print(f"loaded optimizer: {load_path}")

    def save_network(self, network, network_label, epoch_label):
        save_filename = f"net_iter_{epoch_label}_id_{network_label}.pth"
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        print(f"saved net: {save_path}")

    def update_learning_rate(self):
        pass

    def print_network(self, network):
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print(f"Total number of parameters: {num_params}")

    def get_scheduler(self, optimizer, opt):
        if opt.lr_policy == "lambda":
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == "step":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == "plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
        else:
            return NotImplementedError(f"learning rate policy {opt.lr_policy} is not implemented.")
        return scheduler


class FlowCompositionForTrainer(FlowComposition):

    def __init__(self, opt):
        super(FlowCompositionForTrainer, self).__init__(opt)

        smpl = SMPL(model_path=self._opt.smpl_model)
        smpl.eval()
        self.smpl = smpl

    def forward(self, src_img, ref_img, src_smpl, ref_smpl, src_mask=None, ref_mask=None,
                links_ids=None, offsets=0, temporal=False):
        """
        Args:
            src_img (torch.tensor) : (bs, ns, 3, H, W);
            ref_img (torch.tensor) : (bs, nt, 3, H, W);
            src_smpl (torch.tensor): (bs, ns, 85);
            ref_smpl (torch.tesnor): (bs, nt, 85);
            src_mask (torch.tensor): (bs, ns, 3, H, W) or None, front is 0, background is 1;
            ref_mask (torch.tensor): (bs, nt, 3, H, W) or None, front is 0, background is 1;
            links_ids (torch.tensor): (bs, ns + nt, number of verts, 2);
            offsets (torch.tensor) : (bs, nv, 3) or 0;
            temporal (bool): if true, then it will calculate the temporal warping flow, otherwise Ttt will be None

        Returns:
            input_G_bg  (torch.tensor) :  (bs, ns, 4, H, W)
            input_G_src (torch.tensor) :  (bs, ns, 6, H, W)
            input_G_tsf (torch.tensor) :  (bs, nt, 3, H, W)
            Tst         (torch.tensor) :  (bs, nt, ns, H, W, 2)
            Ttt         (torch.tensor) :  (bs, nt - 1, H, W, 2) if temporal is True else return None

        """
        bs, ns, _, h, w = src_img.shape
        bs, nt = ref_img.shape[0:2]

        input_G_bg, input_G_src, input_G_tsf, Tst, Ttt, uv_img, src_info, ref_info = super().forward(
            src_img, ref_img, src_smpl, ref_smpl, src_mask, ref_mask,
            links_ids=links_ids, offsets=offsets, temporal=temporal
        )

        if src_mask is None:
            src_mask = src_info["cond"][:, -1:]
        else:
            src_mask = src_info["masks"]

        if ref_mask is None:
            tsf_mask = ref_info["cond"][:, -1:]
        else:
            tsf_mask = ref_info["masks"]

        src_mask = morph(src_mask, ks=self._opt.ft_ks, mode="erode")
        tsf_mask = morph(tsf_mask, ks=self._opt.ft_ks, mode="erode")

        src_mask = src_mask.view(bs, ns, 1, h, w)
        tsf_mask = tsf_mask.view(bs, nt, 1, h, w)

        head_bbox = self.cal_head_bbox_by_kps(ref_info["j2d"])
        body_bbox = self.cal_body_bbox_by_kps(ref_info["j2d"])

        return input_G_bg, input_G_src, input_G_tsf, Tst, Ttt, src_mask, tsf_mask, head_bbox, body_bbox, uv_img

    @staticmethod
    def cal_head_bbox_by_mask(head_mask, factor=1.2):
        """
        Args:
            head_mask (torch.cuda.FloatTensor): (N, 1, 256, 256).
            factor (float): the factor to enlarge the bbox of head.

        Returns:
            bbox (np.ndarray.int32): (N, 4), hear, 4 = (left_top_x, left_top_y, right_top_x, right_top_y)

        """
        bs, _, height, width = head_mask.shape

        bbox = torch.zeros((bs, 4), dtype=torch.long)

        for i in range(bs):
            mask = head_mask[i, 0]
            coors = (mask == 1).nonzero(as_tuple=False)

            if len(coors) == 0:
                bbox[i, 0] = 0
                bbox[i, 1] = width
                bbox[i, 2] = 0
                bbox[i, 3] = height
            else:
                ys = coors[:, 0]
                xs = coors[:, 1]
                min_y = ys.min()  # left top of Y
                min_x = xs.min()  # left top of X

                max_y = ys.max()  # right top of Y
                max_x = xs.max()  # right top of X

                h = max_y - min_y  # height of head
                w = max_x - min_x  # width of head

                cy = (min_y + max_y) // 2  # (center of y)
                cx = (min_x + max_x) // 2  # (center of x)

                _h = h * factor
                _w = w * factor

                _lt_y = max(0, int(cy - _h / 2))
                _lt_x = max(0, int(cx - _w / 2))

                _rt_y = min(height, int(cy + _h / 2))
                _rt_x = min(width, int(cx + _w / 2))

                bbox[i, 0] = _lt_x
                bbox[i, 1] = _rt_x
                bbox[i, 2] = _lt_y
                bbox[i, 3] = _rt_y

        return bbox

    def cal_head_bbox_by_kps(self, kps):
        """
        Args:
            kps: (N, 19, 2)

        Returns:
            bbox: (N, 4)
        """
        NECK_IDS = 12

        image_size = self._opt.image_size

        kps = (kps + 1) / 2.0

        necks = kps[:, NECK_IDS, 0]
        zeros = torch.zeros_like(necks)
        ones = torch.ones_like(necks)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_x, _ = torch.min(kps[:, NECK_IDS:, 0] - 0.05, dim=1)
        min_x = torch.max(min_x, zeros)

        max_x, _ = torch.max(kps[:, NECK_IDS:, 0] + 0.05, dim=1)
        max_x = torch.min(max_x, ones)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_y, _ = torch.min(kps[:, NECK_IDS:, 1] - 0.05, dim=1)
        min_y = torch.max(min_y, zeros)

        max_y, _ = torch.max(kps[:, NECK_IDS:, 1], dim=1)
        max_y = torch.min(max_y, ones)

        min_x = (min_x * image_size).long()  # (T, 1)
        max_x = (max_x * image_size).long()  # (T, 1)
        min_y = (min_y * image_size).long()  # (T, 1)
        max_y = (max_y * image_size).long()  # (T, 1)

        # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
        rects = torch.stack((min_x, max_x, min_y, max_y), dim=1)
        # import ipdb
        # ipdb.set_trace()
        return rects

    def cal_body_bbox_by_kps(self, kps, factor=1.2):
        """
        Args:
            kps (torch.cuda.FloatTensor): (N, 19, 2)
            factor (float):

        Returns:
            bbox: (N, 4)
        """
        image_size = self._opt.image_size
        bs = kps.shape[0]
        kps = (kps + 1) / 2.0
        zeros = torch.zeros((bs,), device=kps.device)
        ones = torch.ones((bs,), device=kps.device)

        min_x, _ = torch.min(kps[:, :, 0], dim=1)
        max_x, _ = torch.max(kps[:, :, 0], dim=1)
        middle_x = (min_x + max_x) / 2
        width = (max_x - min_x) * factor
        min_x = torch.max(zeros, middle_x - width / 2)
        max_x = torch.min(ones, middle_x + width / 2)

        min_y, _ = torch.min(kps[:, :, 1], dim=1)
        max_y, _ = torch.max(kps[:, :, 1], dim=1)
        middle_y = (min_y + max_y) / 2
        height = (max_y - min_y) * factor
        min_y = torch.max(zeros, middle_y - height / 2)
        max_y = torch.min(ones, middle_y + height / 2)

        min_x = (min_x * image_size).long()  # (T,)
        max_x = (max_x * image_size).long()  # (T,)
        min_y = (min_y * image_size).long()  # (T,)
        max_y = (max_y * image_size).long()  # (T,)

        # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
        bboxs = torch.stack((min_x, max_x, min_y, max_y), dim=1)

        return bboxs


