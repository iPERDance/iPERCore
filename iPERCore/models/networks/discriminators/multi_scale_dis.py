# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .patch_dis import PatchDiscriminator


def reduce_tensor(outs):
    with torch.no_grad():
        avg = 0.0
        num = len(outs)

        for out in outs:
            avg += torch.mean(out)

        avg /= num
        return avg


def crop_img(imgs, rects, fact=2):
    """
    Args:
        imgs (torch.Tensor): (N, C, H, W);
        rects (torch.Tensor): (N, 4=(x0, y0, x1, y1));
        fact (float): the crop and size factor.

    Returns:
        crops (torch.Tensor):

    """
    bs, _, ori_h, ori_w = imgs.shape
    crops = []
    for i in range(bs):
        min_x, max_x, min_y, max_y = rects[i].detach()
        if (min_x != max_x) and (min_y != max_y):
            _crop = imgs[i:i+1, :, min_y:max_y, min_x:max_x]  # (1, c, h", w")
            _crop = F.interpolate(_crop, size=(ori_h // fact, ori_w // fact), mode="bilinear", align_corners=True)
            crops.append(_crop)

    if len(crops) != 0:
        crops = torch.cat(crops, dim=0)

    return crops


class GlobalDiscriminator(nn.Module):

    def __init__(self, cfg, use_aug_bg=False):
        """

        Args:
            cfg (dict or EasyDict): the configurations, and it contains the followings,
                --cond_nc (int): the input dimension;
                --ndf (int): the number of filters at the first layer, default is 64;
                --n_layers (int): the number of downsampling operations, such as the convolution with stride 2, default is 4;
                --max_nf_mult (int): the max factor of ndf, max_nf_mult * ndf, default is 8;
                --norm_type (str): the type of normalization, default is instance;
                --use_sigmoid (bool): use sigmoid or not, default is False.

            use_aug_bg (bool): use aug background or not.
        """

        super(GlobalDiscriminator, self).__init__()
        self.global_model = PatchDiscriminator(
            input_nc=cfg.cond_nc, ndf=cfg.ndf,
            n_layers=cfg.n_layers, max_nf_mult=cfg.max_nf_mult,
            norm_type=cfg.norm_type, use_sigmoid=cfg.use_sigmoid
        )

        if use_aug_bg:
            self.bg_model = PatchDiscriminator(
                input_nc=cfg.bg_cond_nc, ndf=cfg.ndf,
                n_layers=cfg.n_layers, max_nf_mult=cfg.max_nf_mult,
                norm_type=cfg.norm_type, use_sigmoid=cfg.use_sigmoid
            )
        else:
            self.bg_model = None

        self.use_aug_bg = use_aug_bg

    def forward(self, inputs):
        """

        Args:
            inputs (dict): the inputs information, and it contains the followings,
                --x (torch.Tensor): (N, C, H, W);
                --bg_x (torch.Tensor): (N, C, H, W).
        Returns:
            outs (list of torch.Tensor):
        """

        x = inputs["x"]
        bg_x = inputs["bg_x"]
        get_avg = inputs["get_avg"]

        global_outs = self.global_model(x)
        outs = [global_outs]

        if bg_x is not None and self.use_aug_bg:
            bg_outs = self.bg_model(bg_x)
            outs.append(bg_outs)

        if get_avg:
            return outs, reduce_tensor(outs)
        else:
            return outs


class GlobalLocalDiscriminator(nn.Module):
    def __init__(self, cfg, use_aug_bg=False):
        """

        Args:
            cfg (dict or EasyDict): the configurations, and it contains the followings,
                --cond_nc (int): the input dimension;
                --ndf (int): the number of filters at the first layer, default is 64;
                --n_layers (int): the number of downsampling operations, such as the convolution with stride 2, default is 4;
                --max_nf_mult (int): the max factor of ndf, max_nf_mult * ndf, default is 8;
                --norm_type (str): the type of normalization, default is instance;
                --use_sigmoid (bool): use sigmoid or not, default is False.

            use_aug_bg (bool): use aug background or not.
        """

        super(GlobalLocalDiscriminator, self).__init__()

        cond_nc = cfg.cond_nc
        bg_cond_nc = cfg.bg_cond_nc
        ndf = cfg.ndf
        n_layers = cfg.n_layers
        max_nf_mult = cfg.max_nf_mult
        norm_type = cfg.norm_type
        use_sigmoid = cfg.use_sigmoid

        self.global_model = PatchDiscriminator(cond_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                               norm_type=norm_type, use_sigmoid=use_sigmoid)
        self.local_model = PatchDiscriminator(cond_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                              norm_type=norm_type, use_sigmoid=use_sigmoid)

        if use_aug_bg:
            self.bg_model = PatchDiscriminator(
                input_nc=bg_cond_nc, ndf=ndf,
                n_layers=n_layers, max_nf_mult=max_nf_mult,
                norm_type=norm_type, use_sigmoid=use_sigmoid
            )
        else:
            self.bg_model = None

        self.use_aug_bg = use_aug_bg

    def forward(self, inputs):
        """

        Args:
            inputs (dict): the inputs informations, and it contains the followings,
                --x (torch.Tensor): (N, C, H, W);
                --bg_x (torch.Tensor): (N, C, H, W);
                --body_rects (torch.Tensor): (N, 4 = (x0, x1, y0, y1));
                --get_avg (bool): get the avg tensor or not.

        Returns:
            outs (list of torch.Tensor):
        """

        x = inputs["x"]
        bg_x = inputs["bg_x"]
        body_rects = inputs["body_rects"]
        get_avg = inputs["get_avg"]

        outs = []

        # 1. background outputs
        if bg_x is not None and self.use_aug_bg:
            bg_outs = self.bg_model(bg_x)
            outs.append(bg_outs)

        # 2. global outputs
        global_outs = self.global_model(x)
        outs.append(global_outs)

        # 3. local outputs
        crop_imgs = crop_img(x, body_rects, fact=2)
        if len(crop_imgs) != 0:
            local_outs = self.local_model(crop_imgs)
            outs.append(local_outs)

        if get_avg:
            return outs, reduce_tensor(outs)
        else:
            return outs


class GlobalBodyHeadDiscriminator(nn.Module):
    def __init__(self, cfg, use_aug_bg=False):
        """

        Args:
            cfg (dict or EasyDict): the configurations, and it contains the followings,
                --cond_nc (int): the input dimension;
                --ndf (int): the number of filters at the first layer, default is 64;
                --n_layers (int): the number of downsampling operations, such as the convolution with stride 2, default is 4;
                --max_nf_mult (int): the max factor of ndf, max_nf_mult * ndf, default is 8;
                --norm_type (str): the type of normalization, default is instance;
                --use_sigmoid (bool): use sigmoid or not, default is False.

            use_aug_bg (bool): use aug background or not.
        """
        super(GlobalBodyHeadDiscriminator, self).__init__()

        cond_nc = cfg.cond_nc
        bg_cond_nc = cfg.bg_cond_nc
        ndf = cfg.ndf
        n_layers = cfg.n_layers
        max_nf_mult = cfg.max_nf_mult
        norm_type = cfg.norm_type
        use_sigmoid = cfg.use_sigmoid

        self.global_model = PatchDiscriminator(cond_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                               norm_type=norm_type, use_sigmoid=use_sigmoid)
        self.body_model = PatchDiscriminator(cond_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                             norm_type=norm_type, use_sigmoid=use_sigmoid)
        self.head_model = PatchDiscriminator(cond_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                             norm_type=norm_type, use_sigmoid=use_sigmoid)

        if use_aug_bg:
            self.bg_model = PatchDiscriminator(
                input_nc=bg_cond_nc, ndf=ndf,
                n_layers=n_layers, max_nf_mult=max_nf_mult,
                norm_type=norm_type, use_sigmoid=use_sigmoid
            )
        else:
            self.bg_model = None

        self.use_aug_bg = use_aug_bg

    def forward(self, inputs):
        """

        Args:
            inputs (dict): the inputs informations, and it contains the followings,
                --x (torch.Tensor): (N, C, H, W);
                --bg_x (torch.Tensor): (N, C, H, W);
                --body_rects (torch.LongTensor): (N, 4 = (x0, x1, y0, y1));
                --head_rects (torch.LongTensor): (N, 4 = (x0, x1, y0, y1));
                --get_avg (bool):

        Returns:

        """

        x = inputs["x"]
        bg_x = inputs["bg_x"]
        body_rects = inputs["body_rects"]
        head_rects = inputs["head_rects"]
        get_avg = inputs["get_avg"]

        outs = []

        # 1. background outputs
        if bg_x is not None and self.use_aug_bg:
            bg_outs = self.bg_model(bg_x)
            outs.append(bg_outs)

        # 2. global outputs
        global_outs = self.global_model(x)
        outs.append(global_outs)

        # 3. body outputs
        body_imgs = crop_img(x, body_rects, fact=2)
        if len(body_imgs) != 0:
            body_outs = self.body_model(body_imgs)
            outs.append(body_outs)

        # 4. head outputs
        head_imgs = crop_img(x, head_rects, fact=4)
        if len(head_imgs) != 0:
            head_outs = self.head_model(head_imgs)
            outs.append(head_outs)

        if get_avg:
            return outs, reduce_tensor(outs)
        else:
            return outs


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, global_nc, input_nc, ndf=32, n_layers=3, max_nf_mult=8, norm_type="batch", use_sigmoid=False):
        super(MultiScaleDiscriminator, self).__init__()

        # low-res to high-res
        scale_models = nn.ModuleList()
        # scale_n_layers = [1, 1, 1, 1, 1]
        n_scales = 2
        for i in range(n_scales):
            # n_layers = scale_n_layers[i]
            model = PatchDiscriminator(input_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                       norm_type=norm_type, use_sigmoid=use_sigmoid)
            scale_models.append(model)

        self.n_scales = n_scales
        self.scale_models = scale_models

        if global_nc is not None:
            self.global_model = PatchDiscriminator(global_nc, ndf=ndf, n_layers=n_layers, max_nf_mult=max_nf_mult,
                                                   norm_type=norm_type, use_sigmoid=use_sigmoid)
        else:
            self.global_model = None

    def forward(self, global_x, local_x, body_rects, head_rects, get_avg=True):
        scale_outs = []

        if self.global_model is not None:
            outs = self.global_model(global_x)
            scale_outs.append(outs)

        _, _, ori_h, ori_w = local_x.shape
        x = local_x
        for i in range(0, self.n_scales):
            outs = self.scale_models[i](x)

            if i < self.n_scales - 1:
                fact = 2 ** (i + 1)
                x = F.interpolate(local_x, size=(ori_h // fact, ori_w // fact), mode="bilinear", align_corners=True)

            scale_outs.append(outs)

        if get_avg:
            return scale_outs, self.reduce_tensor(scale_outs)
        else:
            return scale_outs
