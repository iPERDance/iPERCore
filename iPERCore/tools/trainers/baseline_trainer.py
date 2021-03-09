# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
from collections import OrderedDict

from iPERCore.tools.utils.filesio.cv_utils import tensor2im

from .lwg_trainer import LWGTrainer


__all__ = ["BaselineTrainer"]


class BaselineTrainer(LWGTrainer):
    """This trainer is used for InputConcat and TextureWarping.
    """

    def optimize_parameters(self, trainable=True, keep_data_for_visuals=False):
        """

        Args:
            trainable:
            keep_data_for_visuals:

        Returns:

        """

        # run inference
        fake_bg, fake_tsf_imgs, fake_masks = self.forward(
            keep_data_for_visuals=keep_data_for_visuals)

        # train G
        if trainable:
            loss_G = self.optimize_G(fake_bg, fake_tsf_imgs, fake_masks)
            self._optimizer_G.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()

        if self._use_gan:
            loss_D = self.optimize_D(fake_bg, fake_tsf_imgs)
            self._optimizer_D.zero_grad()
            loss_D.backward()
            self._optimizer_D.step()

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        """

        Args:
            keep_data_for_visuals (bool):
            return_estimates (bool):

        Returns:

        """

        # generate fake images
        input_G_tsf = self._input_G_tsf
        fake_bg, fake_tsf_color, fake_tsf_mask = \
            self.G(self._input_G_bg, self._input_G_src, input_G_tsf, Tst=self._Tst, Ttt=self._Ttt, only_tsf=False)

        if not self._opt.share_bg:
            fake_bg_tsf = fake_bg[:, 0:1]
        else:
            fake_bg_tsf = fake_bg

        fake_tsf_imgs = fake_tsf_mask * fake_bg_tsf + (1 - fake_tsf_mask) * fake_tsf_color

        # keep data for visualization
        if keep_data_for_visuals:
            self.visual_imgs(fake_bg, fake_tsf_imgs, fake_tsf_mask)

        return fake_bg, fake_tsf_imgs, fake_tsf_mask

    def optimize_G(self, fake_bg, fake_tsf_imgs, fake_masks):
        """

        Args:
            fake_bg (torch.Tensor):
            fake_tsf_imgs (torch.Tensor):
            fake_masks (torch.Tensor):

        Returns:

        """

        ns = self._ns
        bs, nt, c, h, w = fake_tsf_imgs.shape
        fake_bg = fake_bg.view(-1, 3, h, w)
        fake_tsf_imgs = fake_tsf_imgs.view(bs * nt, c, h, w)
        real_tsf_imgs = self._real_tsf.view(bs * nt, -1, h, w)

        if self._use_gan:
            tsf_cond = self._input_G_tsf[:, :, -3:].view(bs * nt, -1, h, w)
            fake_input_D = torch.cat([fake_tsf_imgs, tsf_cond], dim=1)

            # print(fake_bg.shape, self._real_bg.shape)

            d_inputs = {
                "x": fake_input_D,
                "bg_x": None,
                "body_rects": self._body_bbox,
                "head_rects": self._head_bbox,
                "get_avg": False
            }

            # gan loss
            d_fake_outs = self.D(d_inputs)
            self._loss_g_adv = self.crt_gan(d_fake_outs, 0) * self._train_opts.lambda_D_prob

        # perceptual loss
        self._loss_g_rec = self.crt_l1(fake_bg, self._real_bg) * self._train_opts.lambda_rec

        self._loss_g_tsf = self.crt_tsf(fake_tsf_imgs, real_tsf_imgs) * self._train_opts.lambda_tsf

        # face loss
        if self._train_opts.use_face:
            self._loss_g_face = self.crt_face(
                fake_tsf_imgs, real_tsf_imgs,
                bbox1=self._head_bbox, bbox2=self._head_bbox) * self._train_opts.lambda_face

        # mask loss
        real_masks = self._body_mask[:, ns:].view(bs * nt, 1, h, w)
        fake_masks = fake_masks.view(bs * nt, 1, h, w)
        self._loss_g_mask = self.crt_mask(fake_masks, real_masks) * self._train_opts.lambda_mask
        self._loss_g_smooth = self.crt_tv(fake_masks) * self._train_opts.lambda_mask_smooth

        # combine losses
        return self._loss_g_rec + self._loss_g_tsf + self._loss_g_face + \
               self._loss_g_adv + self._loss_g_mask + self._loss_g_smooth

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        # inputs
        visuals["0_source"] = self._vis_source
        visuals["1_uv_img"] = self._vis_uv_img
        visuals["2_real_img"] = self._vis_real
        visuals["3_fake_tsf"] = self._vis_fake_tsf
        visuals["4_fake_bg"] = self._vis_fake_bg
        visuals["5_fake_mask"] = self._vis_mask
        visuals["6_body_mask"] = self._vis_body_mask

        return visuals

    @torch.no_grad()
    def visual_imgs(self, fake_bg, fake_tsf_imgs, fake_masks):
        self._vis_fake_bg = tensor2im(fake_bg[0], idx=-1)
        self._vis_fake_tsf = tensor2im(fake_tsf_imgs[0], idx=-1)
        self._vis_uv_img = tensor2im(self._uv_img, idx=-1)
        self._vis_real = tensor2im(self._real_tsf[0], idx=-1)
        self._vis_source = tensor2im(self._real_src[0], idx=-1)

        ids = self._opt.num_source - 1
        self._vis_mask = tensor2im(fake_masks[0], idx=-1)
        self._vis_body_mask = tensor2im(self._body_mask[0], idx=-1)
