# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import abc
import torch
import torch.nn.functional as F
from collections import OrderedDict

from iPERCore.models.networks import NetworksFactory
from iPERCore.models.networks.criterions import VGGLoss, FaceLoss, LSGANLoss, TVLoss, TemporalSmoothLoss
from iPERCore.tools.utils.filesio.cv_utils import tensor2im

from .base import BaseTrainerModel, FlowCompositionForTrainer

__all__ = ["LWGTrainerABC", "LWGTrainer", "LWGAugBGTrainer", "LWGFrontTrainer"]


class LWGTrainerABC(BaseTrainerModel, abc.ABC):
    """
    The abstract class of LWGTrainer. It implements some basic functions, including initialization all sub-networks,
    gpu_wrapper, loss functions, visualizations, saving and restoring the checkpoints.

    The inherited classes need to implement the followings functions:
        # this function needs to prepare all data for optimization.
        def set_input(self, *inputs):
            pass

        # the forward function
        def forward(self, keep_data_for_visuals):
            pass

        def optimize_G(self):
            pass

        def optimize_D(self):
            pass

    """

    def __init__(self, opt, device):
        """

        Args:
            opt (dict or EasyDict): the option information, and it contains the followings,
                --gpu_ids:
                --image_size (int): the image size;
                --num_source (int): fix the number of sources;
                --time_step (int): the time step for temporal attention;
                --share_bg (bool): whether share background or not, if not, then all images need to use inpainting;

                --Train (dict or EasyDict): the options for training, and it contains the followings,
                    --use_face(bool):
                    --face_factor (float): default is 1.0
                    --face_loss_path (str): default is "./assets/pretrains/sphere20a_20171020.pth";
                    --vgg_type (str): if it is `None`, then we will not use Perceptual Loss;
                    --vgg_loss_path (str):
                    --lambda_rec (float): the loss coefficient for SIDNet, default is 10.0;
                    --lambda_tsf (float): the loss coefficient for TSFNet, default is 10.0
                    --lambda_face (float): the coefficient for face loss, default is 5.0;
                    --lambda_mask (float): the coefficient for mask loss, default is 5.0;
                    --lambda_tv = 1.0
                    --lambda_D_prob (float): the coefficient for gan loss, default is 1.0;
                    --opti (str): the optimization type, default is "Adam";
                    --train_G_every_n_iterations (int): update generator every n iterations, default is 1;
                    --G_adam_b1 (float): 0.9
                    --G_adam_b2 (float): 0.999
                    --D_adam_b1 (float): 0.9
                    --D_adam_b2 (float): 0.999
                    --lr_G (float): 0.0001
                    --lr_D (float): 0.0001
                    --final_lr (float): 0.000002
                    --niters_or_epochs_no_decay (float): 100, fixing learning rate at the first niters_or_epochs_no_decay;
                    --niters_or_epochs_decay (float): 0, then, decreasing the learning rate;
                    --aug_bg (bool): for personalization, it is false; for training, it is true.
            device (torch.device):
        """

        super(LWGTrainerABC, self).__init__(opt)
        self._name = "LWGTrainerABC"

        self.device = device
        self._share_bg = self._opt.share_bg
        self._ns = self._opt.num_source
        self._nt = self._opt.time_step
        self._train_opts = self._opt.Train
        self._aug_bg = self._train_opts.aug_bg
        self._use_gan = self._train_opts.lambda_D_prob > 0

        self._create_network()

        # init train variables and losses
        self._make_optimizer()
        self._init_losses()

        # load networks and optimizers
        if self._opt.load_iter > 0:
            self.load()
        else:
            if self._opt.load_path_G != "None":
                self.load_params(self.G, self._opt.load_path_G, need_module=False)

            if self._opt.load_path_D != "None" and self._use_gan:
                self.load_params(self.D, self._opt.load_path_D, need_module=False)

    def _create_network(self):
        # build flow composition module
        self.flow_comp = FlowCompositionForTrainer(opt=self._opt)
        self.flow_comp.eval()

        # build generator network
        self.G = self._create_generator(cfg=self._opt.neural_render_cfg.Generator)

        # build discriminator network
        if self._use_gan:
            self.D = self._create_discriminator(cfg=self._opt.neural_render_cfg.Discriminator)
        else:
            self.D = None

    def _create_generator(self, cfg):
        """

        Args:
            cfg (dict or EasyDict): the configurations of the generator.

        Returns:

        """

        gen_name = self._opt.gen_name
        return NetworksFactory.get_by_name(gen_name, cfg=cfg, temporal=self._opt.temporal)

    def _create_discriminator(self, cfg):
        dis_name = self._opt.dis_name
        return NetworksFactory.get_by_name(dis_name, cfg=cfg, use_aug_bg=self._aug_bg)

    def _make_optimizer(self):
        self._current_lr_G = self._train_opts.lr_G
        self._current_lr_D = self._train_opts.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self._current_lr_G,
                                             betas=(self._train_opts.G_adam_b1, self._train_opts.G_adam_b2))

        if self._use_gan:
            self._optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self._current_lr_D,
                                                 betas=(self._train_opts.D_adam_b1, self._train_opts.D_adam_b2))

    def _init_losses(self):
        # define loss functions
        self.crt_l1 = torch.nn.L1Loss()
        self.crt_mask = torch.nn.BCELoss()

        if self._train_opts.use_vgg != "None":
            self.crt_tsf = VGGLoss(vgg_type=self._train_opts.use_vgg,
                                   ckpt_path=self._train_opts.vgg_loss_path, resize=True)
        else:
            self.crt_tsf = torch.nn.L1Loss()

        if self._train_opts.use_face:
            self.crt_face = FaceLoss(pretrained_path=self._train_opts.face_loss_path,
                                     factor=self._train_opts.face_factor)

        self.crt_gan = LSGANLoss()
        self.crt_tv = TVLoss()
        self.crt_ts = TemporalSmoothLoss()

        # init losses G
        self._loss_g_rec = 0.0
        self._loss_g_tsf = 0.0
        self._loss_g_adv = 0.0
        self._loss_g_mask = 0.0
        self._loss_g_smooth = 0.0
        self._loss_g_face = 0.0

        # init losses D
        self._d_real = 0.0
        self._d_fake = 0.0

    def multi_gpu_wrapper(self, f):
        self.crt_tsf = self.crt_tsf.to(self.device)
        self.flow_comp = self.flow_comp.to(self.device)
        self.G = f(self.G.to(self.device))

        if self._train_opts.use_face:
            self.crt_face = self.crt_face.to(self.device)

        if self._use_gan:
            self.D = f(self.D.to(self.device))

        return self

    def gpu_wrapper(self):

        self.crt_tsf = self.crt_tsf.to(self.device)
        self.flow_comp = self.flow_comp.to(self.device)
        self.G = self.G.to(self.device)

        if self._train_opts.use_face:
            self.crt_face = self.crt_face.to(self.device)

        if self._use_gan:
            self.D = self.D.to(self.device)

        return self

    def set_train(self):
        self.G.train()

        if self._use_gan:
            self.D.train()

    def set_eval(self):
        self.G.eval()

    def get_current_errors(self):
        loss_g_face = self._loss_g_face if isinstance(self._loss_g_face, float) else self._loss_g_face.item()
        loss_g_smooth = self._loss_g_smooth if isinstance(self._loss_g_smooth, float) else self._loss_g_smooth.item()
        loss_g_mask = self._loss_g_mask if isinstance(self._loss_g_mask, float) else self._loss_g_mask.item()
        loss_g_rec = self._loss_g_rec if isinstance(self._loss_g_rec, float) else self._loss_g_rec.item()
        loss_g_tsf = self._loss_g_tsf if isinstance(self._loss_g_tsf, float) else self._loss_g_tsf.item()
        loss_g_adv = self._loss_g_adv if isinstance(self._loss_g_adv, float) else self._loss_g_adv.item()
        d_real = self._d_real if isinstance(self._d_real, float) else self._d_real.item()
        d_fake = self._d_fake if isinstance(self._d_fake, float) else self._d_fake.item()
        loss_dict = OrderedDict([("g_rec", loss_g_rec),
                                 ("g_tsf", loss_g_tsf),
                                 ("g_face", loss_g_face),
                                 ("g_adv", loss_g_adv),
                                 ("g_mask", loss_g_mask),
                                 ("g_mask_smooth", loss_g_smooth),
                                 ("d_real", d_real),
                                 ("d_fake", d_fake)])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([("lr_G", self._current_lr_G), ("lr_D", self._current_lr_D)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        # inputs
        visuals["0_source"] = self._vis_source
        visuals["1_uv_img"] = self._vis_uv_img
        visuals["2_real_img"] = self._vis_real
        visuals["3_fake_src"] = self._vis_fake_src
        visuals["4_fake_tsf"] = self._vis_fake_tsf
        visuals["5_fake_bg"] = self._vis_fake_bg
        visuals["6_fake_mask"] = self._vis_mask
        visuals["7_body_mask"] = self._vis_body_mask

        # visuals["8_warp_img"] = self._vis_warp
        # visuals["9_src_warp"] = self._vis_src_warp
        # visuals["10_src_ft"] = self._vis_src_ft
        # if self._opt.temporal:
        #     visuals["11_temp_warp"] = self._vis_temp_warp

        return visuals

    @torch.no_grad()
    def visual_imgs(self, fake_bg, fake_src_imgs, fake_tsf_imgs, fake_masks):
        self._vis_fake_bg = tensor2im(fake_bg[0], idx=-1)
        self._vis_fake_src = tensor2im(fake_src_imgs[0], idx=-1)
        self._vis_fake_tsf = tensor2im(fake_tsf_imgs[0], idx=-1)
        self._vis_uv_img = tensor2im(self._uv_img, idx=-1)
        self._vis_real = tensor2im(self._real_tsf[0], idx=-1)
        self._vis_source = tensor2im(self._real_src[0], idx=-1)

        ids = self._opt.num_source - 1
        self._vis_mask = tensor2im(fake_masks[0, ids:], idx=-1)
        self._vis_body_mask = tensor2im(self._body_mask[0, ids:], idx=-1)

        # self._vis_warp = tensor2im(self._input_G_tsf[0, :, 0:3], idx=-1)
        # self._vis_src_warp = tensor2im(F.grid_sample(self._input_G_src[0, :, 0:3], self._Tst[0, 0]), idx=-1)
        # self._vis_src_ft = tensor2im(self._input_G_src[0, :, 0:3])
        # if self._opt.temporal:
        #     self._vis_temp_warp = tensor2im(F.grid_sample(fake_tsf_imgs[0, 0:-1], self._Ttt[0]), idx=-1)

    def save(self, label):
        # save networks

        if "module" in self.G.__dict__:
            self.save_network(self.G.module, "G", label)

            if self._use_gan:
                self.save_network(self.D.module, "D", label)
        else:
            self.save_network(self.G, "G", label)

            if self._use_gan:
                self.save_network(self.D, "D", label)

        # save optimizers
        self.save_optimizer(self._optimizer_G, "G", label)

        if self._use_gan:
            self.save_optimizer(self._optimizer_D, "D", label)

    def load(self):
        load_iter = self._opt.load_iter

        # load G
        self.load_network(self.G, "G", load_iter, need_module=False)

        if self._use_gan:
            # load D
            self.load_network(self.D, "D", load_iter, need_module=False)

    def update_learning_rate(self):
        # updated learning rate G
        final_lr = self._train_opts.final_lr

        lr_decay_G = (self._train_opts.lr_G - final_lr) / self._train_opts.niters_or_epochs_decay
        self._current_lr_G -= lr_decay_G
        for param_group in self._optimizer_G.param_groups:
            param_group["lr"] = self._current_lr_G
        print("update G learning rate: %f -> %f" % (self._current_lr_G + lr_decay_G, self._current_lr_G))

        if self._use_gan:
            # update learning rate D
            lr_decay_D = (self._train_opts.lr_D - final_lr) / self._train_opts.niters_or_epochs_decay
            self._current_lr_D -= lr_decay_D
            for param_group in self._optimizer_D.param_groups:
                param_group["lr"] = self._current_lr_D
            print("update D learning rate: %f -> %f" % (self._current_lr_D + lr_decay_D, self._current_lr_D))

    def optimize_parameters(self, trainable=True, keep_data_for_visuals=False):
        """

        Args:
            trainable:
            keep_data_for_visuals:

        Returns:

        """

        # run inference
        fake_bg, fake_src_imgs, fake_tsf_imgs, fake_masks = self.forward(
            keep_data_for_visuals=keep_data_for_visuals)

        # train G
        if trainable:
            loss_G = self.optimize_G(fake_bg, fake_src_imgs, fake_tsf_imgs, fake_masks)
            self._optimizer_G.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()

        if self._use_gan:
            loss_D = self.optimize_D(fake_bg, fake_tsf_imgs)
            self._optimizer_D.zero_grad()
            loss_D.backward()
            self._optimizer_D.step()

    @abc.abstractmethod
    def forward(self, keep_data_for_visuals):
        pass

    @abc.abstractmethod
    def optimize_G(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def optimize_D(self, *args, **kwargs):
        pass


class LWGAugBGTrainer(LWGTrainerABC):
    r"""
    The LWGTrainer with background augmentation. In this class, there are additional background images from
    Place2 dataset. We sample an image I from the Place2 dataset, and then paste a mask on I, and we will get
    an incomplete image, \hat{I}. The BGNet takes the incomplete image, \hat{I} as the inputs, and results in
    an complete image, \hat{I}_{bg}. Since we have the actual background image I, we could thereby learn the BGNet
    in a supervised way. This will improve the generalization of the BGNet.

    In set_input(self, inputs, device), the inputs must provide the "bg".

    """

    def __init__(self, opt, device):
        super().__init__(opt, device)
        self._name = "LWGAugTrainer"

    def set_input(self, inputs, device):
        """

        Args:
            inputs (dict): the inputs information get from the dataset, it contains the following items,
                --images (torch.Tensor): ();
                --smpls (torch.Tensor): ();
                --masks (torch.Tensor): (), the front is 0, and the background is 1;
                --offsets (torch.Tensor): ();
                --links_ids (torch.Tensor): ();

            device (torch.device): e.g. torch.device("cuda:0").

        Returns:
            None
        """

        with torch.no_grad():
            images = inputs["images"].to(device, non_blocking=True)
            aug_bg = inputs["bg"].to(device, non_blocking=True)
            smpls = inputs["smpls"].to(device, non_blocking=True)
            masks = inputs["masks"].to(device, non_blocking=True)
            offsets = inputs["offsets"].to(device, non_blocking=True) if "offsets" in inputs else 0
            links_ids = inputs["links_ids"].to(device, non_blocking=True) if "links_ids" in inputs else None

            ns = self._ns
            src_img = images[:, 0:ns].contiguous()
            src_smpl = smpls[:, 0:ns].contiguous()
            tsf_img = images[:, ns:].contiguous()
            tsf_smpl = smpls[:, ns:].contiguous()
            src_mask = masks[:, 0:ns].contiguous()
            ref_mask = masks[:, ns:].contiguous()

            # print(links_ids.shape, images.shape, smpls.shape, src_img.shape, src_smpl.shape,
            #       tsf_img.shape, tsf_smpl.shape)

            ##################################################################
            # input_G_bg (): for background inpainting network,
            # input_G_src (): for source identity network,
            # input_G_tsf (): for transfer network,
            # Tst ():  the transformation flows from source (s_i) to target (t_j);
            # Ttt ():  if temporal is True, transformation from last time target (t_{j-1)
            #          to current time target (t_j), otherwise, it is None.
            #
            # src_mask (): the source masks;
            # tsf_mask (): the target masks;
            # head_bbox (): the head bounding boxes of all targets;
            # body_bbox (): the body bounding boxes of all targets;
            # uv_img (): the extracted uv images, for visualization.
            ################################################################
            input_G_bg, input_G_src, input_G_tsf, Tst, Ttt, src_mask, tsf_mask, head_bbox, body_bbox, uv_img = \
                self.flow_comp(src_img, tsf_img, src_smpl, tsf_smpl, src_mask=src_mask, ref_mask=ref_mask,
                               links_ids=links_ids, offsets=offsets, temporal=self._opt.temporal)

            self._real_src = src_img
            self._real_tsf = tsf_img

            self._head_bbox = head_bbox
            self._body_bbox = body_bbox
            self._body_mask = masks
            # self._body_mask = torch.cat([src_mask, tsf_mask], dim=1)  # the front is 0, and the background is 1

            self._uv_img = uv_img
            self._Tst = Tst
            self._Ttt = Ttt

            self._input_G_src = input_G_src
            self._input_G_tsf = input_G_tsf

            # if do not share background, here we need to inpaint all images, including the sources and the targets.
            if not self._share_bg:
                input_G_bg_tsf = torch.cat([tsf_img * tsf_mask, tsf_mask], dim=2)
                input_G_bg = torch.cat([input_G_bg, input_G_bg_tsf], dim=1)

            # additional augmented background images.
            src_mask = src_mask[:, 0:1]
            inpug_G_aug_bg = torch.cat([aug_bg[:, None] * src_mask, src_mask], dim=2)
            input_G_bg = torch.cat([input_G_bg, inpug_G_aug_bg], dim=1)
            self._real_bg = aug_bg

            # if self._share_bg is True, input_G_bg = [input_G_bg_src, input_G_tsf, input_G_aug_bg]
            # otherwise, input_G_bg = [input_G_bg_src[:, np.random.randint(0, ns), ...], input_G_aug_bg]
            self._input_G_bg = input_G_bg

    def forward(self, keep_data_for_visuals=False):
        """

        Args:
            keep_data_for_visuals:

        Returns:

        """

        #################################### generate fake images ########################
        # fake_bg ():
        # fake_src_color ():
        # fake_src_mask ():
        # fake_tsf_color ():
        # fake_tsf_mask()
        #
        #################################################################################
        fake_bg, fake_src_color, fake_src_mask, fake_tsf_color, fake_tsf_mask = self.G(
            self._input_G_bg, self._input_G_src, self._input_G_tsf, Tst=self._Tst, Ttt=self._Ttt, only_tsf=False)

        if not self._opt.share_bg:
            fake_bg_src = fake_bg[:, 0:self._ns]
            fake_bg_tsf = fake_bg[:, self._ns:self._ns + self._nt]
        else:
            fake_bg_src = fake_bg[:, 0:1]
            fake_bg_tsf = fake_bg_src

        fake_src_imgs = fake_src_mask * fake_bg_src + (1 - fake_src_mask) * fake_src_color
        fake_tsf_imgs = fake_tsf_mask * fake_bg_tsf + (1 - fake_tsf_mask) * fake_tsf_color
        fake_masks = torch.cat([fake_src_mask, fake_tsf_mask], dim=1)

        # keep data for visualization
        if keep_data_for_visuals:
            self.visual_imgs(fake_bg, fake_src_imgs, fake_tsf_imgs, fake_masks)

        return fake_bg, fake_src_imgs, fake_tsf_imgs, fake_masks

    def optimize_G(self, fake_bg, fake_src_imgs, fake_tsf_imgs, fake_masks):
        """

        Args:
            fake_bg (torch.Tensor):
            fake_src_imgs (torch.Tensor):
            fake_tsf_imgs (torch.Tensor):
            fake_masks (torch.Tensor):

        Returns:

        """

        ns = fake_src_imgs.shape[1]
        bs, nt, c, h, w = fake_tsf_imgs.shape
        fake_tsf_imgs = fake_tsf_imgs.view(bs * nt, c, h, w)
        real_tsf_imgs = self._real_tsf.view(bs * nt, -1, h, w)
        tsf_cond = self._input_G_tsf[:, :, -3:].view(bs * nt, -1, h, w)
        fake_input_D = torch.cat([fake_tsf_imgs, tsf_cond], dim=1)

        fake_aug_bg = fake_bg[:, -1]
        fake_global = torch.cat([fake_aug_bg, self._input_G_bg[:, -1, -1:]], dim=1)

        d_inputs = {
            "x": fake_input_D,
            "bg_x": fake_global,
            "body_rects": self._body_bbox,
            "head_rects": self._head_bbox,
            "get_avg": False
        }

        # gan loss
        d_fake_outs = self.D(d_inputs)
        self._loss_g_adv = self.crt_gan(d_fake_outs, 0) * self._train_opts.lambda_D_prob

        # perceptual loss
        self._loss_g_rec = (self.crt_l1(fake_src_imgs, self._real_src) +
                            self.crt_l1(fake_aug_bg, self._real_bg)) / 2 * self._train_opts.lambda_rec
        self._loss_g_tsf = self.crt_tsf(fake_tsf_imgs, real_tsf_imgs) * self._train_opts.lambda_tsf

        # face loss
        if self._train_opts.use_face:
            self._loss_g_face = self.crt_face(
                fake_tsf_imgs, real_tsf_imgs,
                bbox1=self._head_bbox, bbox2=self._head_bbox) * self._train_opts.lambda_face

        # mask loss
        fake_masks = fake_masks.view(bs * (ns + nt), 1, h, w)
        body_masks = self._body_mask.view(bs * (ns + nt), 1, h, w)
        self._loss_g_mask = self.crt_mask(fake_masks, body_masks) * self._train_opts.lambda_mask
        self._loss_g_smooth = self.crt_tv(fake_masks) * self._train_opts.lambda_mask_smooth

        # combine losses
        return self._loss_g_rec + self._loss_g_tsf + self._loss_g_face + \
               self._loss_g_adv + self._loss_g_mask + self._loss_g_smooth

    def optimize_D(self, fake_bg, fake_tsf_imgs):
        """

        Args:
            fake_bg:
            fake_tsf_imgs:

        Returns:

        """

        bs, nt, c, h, w = fake_tsf_imgs.shape
        fake_tsf_imgs = fake_tsf_imgs.view(bs * nt, c, h, w)
        real_tsf_imgs = self._real_tsf.reshape(bs * nt, c, h, w)

        tsf_cond = self._input_G_tsf[:, :, -3:].view(bs * nt, -1, h, w)
        fake_input_D = torch.cat([fake_tsf_imgs.detach(), tsf_cond], dim=1)
        real_input_D = torch.cat([real_tsf_imgs, tsf_cond], dim=1)

        fake_aug_bg = fake_bg[:, -1]
        fake_bg_x = torch.cat([fake_aug_bg.detach(), self._input_G_bg[:, -1, -1:]], dim=1)
        real_bg_x = torch.cat([self._real_bg, self._input_G_bg[:, -1, -1:]], dim=1)

        real_inputs = {
            "x": real_input_D,
            "bg_x": real_bg_x,
            "body_rects": self._body_bbox,
            "head_rects": self._head_bbox,
            "get_avg": True
        }
        d_real_outs, self._d_real = self.D(real_inputs)

        fake_inputs = {
            "x": fake_input_D,
            "bg_x": fake_bg_x,
            "body_rects": self._body_bbox,
            "head_rects": self._head_bbox,
            "get_avg": True
        }
        d_fake_outs, self._d_fake = self.D(fake_inputs)

        _loss_d_real = self.crt_gan(d_real_outs, 1)
        _loss_d_fake = self.crt_gan(d_fake_outs, -1)

        # combine losses
        return _loss_d_real + _loss_d_fake


class LWGTrainer(LWGTrainerABC):
    """
    This is the LWGTrainer for personalization. In this class, the `inputs` in self.set_input(inputs, device)
    provides a `bg`. We use it to fine-tune the BGNet for a better background inpainting result.
    However, in the most cases, there are not actual background images provided.
    Here, in the Preprocessing stage, we firstly use a pre-trained deepfillv2 on the Place2 dataset to produce
    a inpainted result as the pseudo-background image. Then, we use this pseudo-background image the `ground-truth`
    to regularize the BGNet. In our experiments, we found this trick indeed help the BGNet generate a decent inpainted
    background, since it combines the advantages of both deep image prior and deepfillv2.
    """

    def __init__(self, opt, device):
        super(LWGTrainer, self).__init__(opt, device)
        self._name = "LWGTrainer"

    def set_input(self, inputs, device):
        """

        Args:
            inputs (dict): the inputs information get from the dataset, it contains the following items,
                --images (torch.Tensor): ();
                --smpls (torch.Tensor): ();
                --masks (torch.Tensor): ();
                --offsets (torch.Tensor): ();
                --links_ids (torch.Tensor): ();

            device:

        Returns:

        """

        with torch.no_grad():
            images = inputs["images"].to(device, non_blocking=True)
            bg = inputs["bg"].to(device, non_blocking=True)
            smpls = inputs["smpls"].to(device, non_blocking=True)
            masks = inputs["masks"].to(device, non_blocking=True)
            offsets = inputs["offsets"].to(device, non_blocking=True)
            links_ids = inputs["links_ids"].to(device, non_blocking=True) if "links_ids" in inputs else None

            ns = self._ns

            src_img = images[:, 0:ns].contiguous()
            src_smpl = smpls[:, 0:ns].contiguous()
            tsf_img = images[:, ns:].contiguous()
            tsf_smpl = smpls[:, ns:].contiguous()
            src_mask = masks[:, 0:ns].contiguous()
            ref_mask = masks[:, ns:].contiguous()

            # print(links_ids.shape, images.shape, smpls.shape, src_img.shape, src_smpl.shape,
            #       tsf_img.shape, tsf_smpl.shape)

            ##################################################################
            # input_G_bg (): for background inpainting network,
            # input_G_src (): for source identity network,
            # input_G_tsf (): for transfer network,
            # Tst ():
            # Ttt ():
            # src_mask ():
            # tsf_mask ():
            # head_bbox ():
            # body_bbox ():
            # uv_img ():
            ################################################################

            input_G_bg, input_G_src, input_G_tsf, Tst, Ttt, src_mask, tsf_mask, head_bbox, body_bbox, uv_img = \
                self.flow_comp(src_img, tsf_img, src_smpl, tsf_smpl, src_mask=src_mask, ref_mask=ref_mask,
                               links_ids=links_ids, offsets=offsets, temporal=self._opt.temporal)

            self._real_src = src_img
            self._real_tsf = tsf_img

            self._head_bbox = head_bbox
            self._body_bbox = body_bbox
            self._body_mask = masks

            self._uv_img = uv_img
            self._Tst = Tst
            self._Ttt = Ttt

            self._input_G_src = input_G_src
            self._input_G_tsf = input_G_tsf

            if not self._share_bg:
                input_G_bg_tsf = torch.cat([tsf_img * tsf_mask, tsf_mask], dim=2)
                input_G_bg = torch.cat([input_G_bg, input_G_bg_tsf], dim=1)

            self._input_G_bg = input_G_bg
            self._real_bg = bg.view(-1, 3, self._opt.image_size, self._opt.image_size)

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        """

        Args:
            keep_data_for_visuals (bool):
            return_estimates (bool):

        Returns:

        """

        # generate fake images
        input_G_tsf = self._input_G_tsf
        fake_bg, fake_src_color, fake_src_mask, fake_tsf_color, fake_tsf_mask = \
            self.G(self._input_G_bg, self._input_G_src, input_G_tsf, Tst=self._Tst, Ttt=self._Ttt, only_tsf=False)

        if not self._opt.share_bg:
            fake_bg_src = fake_bg[:, 0:self._ns]
            fake_bg_tsf = fake_bg[:, self._ns:self._ns + self._nt]
        else:
            fake_bg_src = fake_bg
            fake_bg_tsf = fake_bg

        fake_src_imgs = fake_src_mask * fake_bg_src + (1 - fake_src_mask) * fake_src_color
        fake_tsf_imgs = fake_tsf_mask * fake_bg_tsf + (1 - fake_tsf_mask) * fake_tsf_color
        fake_masks = torch.cat([fake_src_mask, fake_tsf_mask], dim=1)

        # keep data for visualization
        if keep_data_for_visuals:
            self.visual_imgs(fake_bg, fake_src_imgs, fake_tsf_imgs, fake_masks)

        return fake_bg, fake_src_imgs, fake_tsf_imgs, fake_masks

    def optimize_G(self, fake_bg, fake_src_imgs, fake_tsf_imgs, fake_masks):
        """

        Args:
            fake_bg (torch.Tensor):
            fake_src_imgs (torch.Tensor):
            fake_tsf_imgs (torch.Tensor):
            fake_masks (torch.Tensor):

        Returns:

        """

        ns = fake_src_imgs.shape[1]
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
        self._loss_g_rec = (self.crt_l1(fake_src_imgs, self._real_src) +
                            self.crt_l1(fake_bg, self._real_bg)) / 2 * self._train_opts.lambda_rec

        self._loss_g_tsf = self.crt_tsf(fake_tsf_imgs, real_tsf_imgs) * self._train_opts.lambda_tsf

        # face loss
        if self._train_opts.use_face:
            self._loss_g_face = self.crt_face(
                fake_tsf_imgs, real_tsf_imgs,
                bbox1=self._head_bbox, bbox2=self._head_bbox) * self._train_opts.lambda_face

        # mask loss
        fake_masks = fake_masks.view(bs * (ns + nt), 1, h, w)
        body_masks = self._body_mask.view(bs * (ns + nt), 1, h, w)
        self._loss_g_mask = self.crt_mask(fake_masks, body_masks) * self._train_opts.lambda_mask
        self._loss_g_smooth = self.crt_tv(fake_masks) * self._train_opts.lambda_mask_smooth

        # combine losses
        return self._loss_g_rec + self._loss_g_tsf + self._loss_g_face + \
               self._loss_g_adv + self._loss_g_mask + self._loss_g_smooth

    def optimize_D(self, fake_bg, fake_tsf_imgs):
        """

        Args:
            fake_bg (torch.Tensor):
            fake_tsf_imgs (torch.Tensor):

        Returns:

        """

        bs, nt, c, h, w = fake_tsf_imgs.shape
        fake_tsf_imgs = fake_tsf_imgs.view(bs * nt, c, h, w)
        real_tsf_imgs = self._real_tsf.reshape(bs * nt, c, h, w)

        tsf_cond = self._input_G_tsf[:, :, -3:].view(bs * nt, -1, h, w)
        fake_input_D = torch.cat([fake_tsf_imgs.detach(), tsf_cond], dim=1)
        real_input_D = torch.cat([real_tsf_imgs, tsf_cond], dim=1)

        real_inputs = {
            "x": real_input_D,
            "bg_x": None,
            "body_rects": self._body_bbox,
            "head_rects": self._head_bbox,
            "get_avg": True
        }
        d_real_outs, self._d_real = self.D(real_inputs)

        fake_inputs = {
            "x": fake_input_D,
            "bg_x": None,
            "body_rects": self._body_bbox,
            "head_rects": self._head_bbox,
            "get_avg": True
        }
        d_fake_outs, self._d_fake = self.D(fake_inputs)

        _loss_d_real = self.crt_gan(d_real_outs, 1)
        _loss_d_fake = self.crt_gan(d_fake_outs, -1)

        # combine losses
        return _loss_d_real + _loss_d_fake


class LWGFrontTrainer(LWGTrainer):
    """
    The LWGTrainer with only frontal parts, and it only trains the SIDNet and the TSFNet, and ignores the BGNet.
    In this class, the `inputs` in self.set_input(inputs, device) must provide a `bg` which is the real background.
    """

    def __init__(self, opt, device):
        super().__init__(opt, device)

    def _create_generator(self, cfg):
        """

        Args:
            cfg (dict or EasyDict): the configurations of the generator.

        Returns:

        """

        return NetworksFactory.get_by_name("AttLWB-Front-SPADE", cfg=cfg, temporal=self._opt.temporal)

    def set_input(self, inputs, device):
        with torch.no_grad():
            images = inputs["images"].to(device, non_blocking=True)
            smpls = inputs["smpls"].to(device, non_blocking=True)
            masks = inputs["masks"].to(device, non_blocking=True)
            offsets = inputs["offsets"].to(device, non_blocking=True)
            bg_imgs = inputs["bg"].to(device, non_blocking=True)
            bg_imgs.unsqueeze_(dim=1)

            ns = self._opt.num_source

            src_img = images[:, 0:ns].contiguous()
            src_smpl = smpls[:, 0:ns].contiguous()
            tsf_img = images[:, ns:].contiguous()
            tsf_smpl = smpls[:, ns:].contiguous()
            src_mask = masks[:, 0:ns]
            ref_mask = masks[:, ns:]

            _, input_G_src, input_G_tsf, Tst, Ttt, src_mask, tsf_mask, head_bbox, body_bbox, uv_img = \
                self.flow_comp(src_img, tsf_img, src_smpl, tsf_smpl, src_mask=src_mask, ref_mask=ref_mask,
                               offsets=offsets, temporal=self._opt.temporal)

            self._real_src = src_img
            self._real_tsf = tsf_img

            self._head_bbox = head_bbox
            self._body_bbox = body_bbox
            # self._body_mask = torch.cat((src_mask, tsf_mask), dim=1)
            self._body_mask = masks

            self._uv_img = uv_img
            self._Tst = Tst
            self._Ttt = Ttt

            self._input_G_src = input_G_src
            self._input_G_tsf = input_G_tsf

            self._bg_imgs = bg_imgs

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        # generate fake images
        input_G_tsf = self._input_G_tsf
        fake_src_color, fake_src_mask, fake_tsf_color, fake_tsf_mask = \
            self.G(self._input_G_src, input_G_tsf, Tst=self._Tst, Ttt=self._Ttt, only_tsf=False)

        fake_src_imgs = fake_src_mask * self._bg_imgs + (1 - fake_src_mask) * fake_src_color
        fake_tsf_imgs = fake_tsf_mask * self._bg_imgs + (1 - fake_tsf_mask) * fake_tsf_color
        fake_masks = torch.cat([fake_src_mask, fake_tsf_mask], dim=1)

        # keep data for visualization
        if keep_data_for_visuals:
            self.visual_imgs(self._bg_imgs, fake_tsf_imgs, fake_masks)

        return fake_src_imgs, fake_tsf_imgs, fake_masks

    def _optimize_G(self, fake_src_imgs, fake_tsf_imgs, fake_masks):
        ns = fake_src_imgs.shape[1]
        bs, nt, c, h, w = fake_tsf_imgs.shape
        fake_tsf_imgs = fake_tsf_imgs.view(bs * nt, c, h, w)
        real_tsf_imgs = self._real_tsf.view(bs * nt, -1, h, w)
        tsf_cond = self._input_G_tsf[:, :, -3:].view(bs * nt, -1, h, w)
        fake_input_D = torch.cat([fake_tsf_imgs, tsf_cond], dim=1)

        d_inputs = {
            "x": fake_input_D,
            "bg_x": None,
            "body_rects": self._body_bbox,
            "head_rects": self._head_bbox,
            "get_avg": False
        }

        d_fake_outs = self.D(d_inputs)

        self._loss_g_adv = self.crt_gan(d_fake_outs, 0) * self._opt.lambda_D_prob

        self._loss_g_rec = self.crt_l1(fake_src_imgs, self._real_src) * self._opt.lambda_rec

        self._loss_g_tsf = self.crt_tsf(fake_tsf_imgs, real_tsf_imgs) * self._opt.lambda_tsf

        if self._opt.use_face:
            self._loss_g_face = self.crt_face(
                fake_tsf_imgs, real_tsf_imgs, bbox1=self._head_bbox, bbox2=self._head_bbox) * self._opt.lambda_face

        # loss mask
        fake_masks = fake_masks.view(bs * (ns + nt), 1, h, w)
        body_masks = self._body_mask.view(bs * (ns + nt), 1, h, w)
        self._loss_g_mask = self.crt_mask(fake_masks, body_masks) * self._opt.lambda_mask
        self._loss_g_smooth = self.crt_tv(fake_masks) * self._opt.lambda_mask_smooth

        # combine losses
        return self._loss_g_rec + self._loss_g_tsf + self._loss_g_face + \
               self._loss_g_adv + self._loss_g_mask + self._loss_g_smooth

    def _optimize_D(self, fake_tsf_imgs):
        bs, nt, c, h, w = fake_tsf_imgs.shape
        fake_tsf_imgs = fake_tsf_imgs.view(bs * nt, c, h, w)
        real_tsf_imgs = self._real_tsf.reshape(bs * nt, c, h, w)
        tsf_cond = self._input_G_tsf[:, :, -3:].view(bs * nt, -1, h, w)
        fake_input_D = torch.cat([fake_tsf_imgs.detach(), tsf_cond], dim=1)
        real_input_D = torch.cat([real_tsf_imgs, tsf_cond], dim=1)

        real_inputs = {
            "x": real_input_D,
            "bg_x": None,
            "body_rects": self._body_bbox,
            "head_rects": self._head_bbox,
            "get_avg": True
        }

        fake_inputs = {
            "x": fake_input_D,
            "bg_x": None,
            "body_rects": self._body_bbox,
            "head_rects": self._head_bbox,
            "get_avg": True
        }

        d_real_outs, self._d_real = self.D(real_inputs)
        d_fake_outs, self._d_fake = self.D(fake_inputs)

        _loss_d_real = self.crt_gan(d_real_outs, 1)
        _loss_d_fake = self.crt_gan(d_fake_outs, -1)

        # combine losses
        return _loss_d_real + _loss_d_fake
