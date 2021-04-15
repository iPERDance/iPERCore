# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC

from .bg_inpaintor import ResNetInpaintor


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.main(x)


class LWB(nn.Module):
    def __init__(self):
        super(LWB, self).__init__()

    def forward(self, X, T):
        """
        Args:
            X (torch.tensor): (N, C, H, W) or (N, nt, C, H, W) or (N, ns, C, H, W)
            T (torch.tensor): (N, h, w, 2) or (N, nt, h, w, 2) or (N, nt, ns, h, w, 2)

        Returns:
            x_warp (torch.tensor): (N, C, H ,W)
        """
        x_shape = X.shape
        T_shape = T.shape
        x_n_dim = len(x_shape)
        T_n_dim = len(T_shape)

        assert x_n_dim >= 4 and T_n_dim >= 4

        if x_n_dim == 4 and T_n_dim == 4:
            warp = self.transform(X, T)

        elif x_n_dim == 5 and T_n_dim == 5:
            bs, nt, C, H, W = x_shape
            h, w = T_shape[2:4]
            warp = self.transform(X.view(bs * nt, C, H, W), T.view(bs * nt, h, w, 2))

        else:
            raise ValueError("#dim of X must >= 4 and #dim of T must >= 4")

        return warp

    def resize_trans(self, x, T):
        _, _, h, w = x.shape

        T_scale = T.permute(0, 3, 1, 2)  # (bs, 2, h, w)
        T_scale = F.interpolate(T_scale, size=(h, w), mode="bilinear", align_corners=True)
        T_scale = T_scale.permute(0, 2, 3, 1)  # (bs, h, w, 2)

        return T_scale

    def transform(self, x, T):
        bs, c, h_x, w_x = x.shape
        bs, h_t, w_t, _ = T.shape

        if h_t != h_x or w_t != w_x:
            T = self.resize_trans(x, T)
        x_trans = F.grid_sample(x, T)
        return x_trans


class AddLWB(nn.Module):
    def __init__(self):
        super().__init__()

        self.lwb = LWB()

    def forward(self, tsf_x, src_x, Tst):
        """

        Args:
            tsf_x  (torch.Tensor): (bs, c, h, w)
            src_x  (torch.Tensor): (bs * ns, c, h, w)
            Tst    (torch.Tensor): (bs * ns, h, w, 2)

        Returns:
            fused_x (torch.Tensor): (bs, c, h, w)
        """

        bsns, _, _, _ = Tst.shape
        bs, c, h, w = tsf_x.shape
        ns = bsns // bs

        warp_x = self.lwb(src_x, Tst).view(bs, ns, -1, h, w)

        # tsf_x.unsqueeze_(dim=1)
        tsf_x = tsf_x.unsqueeze(dim=1)

        # sum
        fused_x = torch.sum(torch.cat([tsf_x, warp_x], dim=1), dim=1)

        return fused_x

    def __str__(self):
        return "AddLWB"

    def __repr__(self):
        return "AddLWB"


class AvgLWB(nn.Module):
    def __init__(self):
        super().__init__()

        self.lwb = LWB()

    def forward(self, tsf_x, src_x, Tst):
        """

        Args:
            tsf_x  (torch.Tensor): (bs, c, h, w)
            src_x  (torch.Tensor): (bs * ns, c, h, w)
            Tst    (torch.Tensor): (bs * ns, h, w, 2)

        Returns:
            fused_x (torch.Tensor): (bs, c, h, w)
        """

        bsns, _, _, _ = Tst.shape
        bs, c, h, w = tsf_x.shape
        ns = bsns // bs

        warp_x = self.lwb(src_x, Tst).view(bs, ns, -1, h, w)

        # tsf_x.unsqueeze_(dim=1)
        tsf_x = tsf_x.unsqueeze(dim=1)

        # mean
        fused_x = torch.mean(torch.cat([tsf_x, warp_x], dim=1), dim=1)

        return fused_x

    def __str__(self):
        return "AvgLWB"

    def __repr__(self):
        return "AvgLWB"


class Encoder(nn.Module):
    def __init__(self, in_channel, num_filters, use_bias=True):
        super().__init__()

        layers = list()
        # Down-Sampling
        self.n_down = len(num_filters)
        for i in range(self.n_down):
            if i == 0:
                c_in = in_channel
            else:
                c_in = num_filters[i - 1]

            block = nn.Sequential(
                nn.Conv2d(c_in, num_filters[i], kernel_size=3, stride=2, padding=1, bias=use_bias),
                nn.ReLU(inplace=True)
            )

            layers.append(block)

        self.layers = nn.Sequential(*layers)

    def forward(self, x, get_details=True):
        if get_details:
            x_list = []
            for i in range(self.n_down):
                x = self.layers[i](x)
                x_list.append(x)

            outs = x_list
        else:
            outs = self.layers(x)

        return outs


class Decoder(nn.Module):
    def __init__(self, in_channel, num_filters):
        super().__init__()

        layers = list()

        self.n_down = len(num_filters)
        for i in range(0, self.n_down):
            if i == 0:
                c_in = in_channel
            else:
                c_in = num_filters[i - 1]

            block = nn.Sequential(
                nn.ConvTranspose2d(c_in, num_filters[i], kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
            layers.append(block)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SkipDecoder(nn.Module):
    def __init__(self, in_channel, enc_num_filters, dec_num_filters):
        super().__init__()

        upconvs = list()
        skippers = list()

        self.n_down = len(dec_num_filters)
        for i in range(0, self.n_down):
            if i == 0:
                d_in = in_channel
            else:
                d_in = dec_num_filters[i - 1]

            upconvs.append(nn.Sequential(
                nn.ConvTranspose2d(d_in, dec_num_filters[i], kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ))

            if i != self.n_down - 1:
                s_in = enc_num_filters[self.n_down - 2 - i] + dec_num_filters[i]
                skippers.append(nn.Sequential(
                    nn.Conv2d(s_in, dec_num_filters[i], kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True)
                ))

        self.skippers = nn.Sequential(*skippers)
        self.upconvs = nn.Sequential(*upconvs)

        # print(self.skippers)
        # print(self.upconvs)

    def forward(self, x, enc_outs):
        d_out = x
        for i in range(self.n_down):
            d_out = self.upconvs[i](d_out)
            if i != self.n_down - 1:
                skip = torch.cat([enc_outs[self.n_down - 2 - i], d_out], dim=1)
                # print(skip.shape, self.skippers[i])
                d_out = self.skippers[i](skip)

        return d_out


class ResAutoEncoder(nn.Module):
    def __init__(self, in_channel=6, num_filters=(64, 128, 128, 128), n_res_block=4):
        super(ResAutoEncoder, self).__init__()
        self._name = "ResAutoEncoder"

        # build encoders
        self.encoders = Encoder(in_channel=in_channel, num_filters=num_filters)

        res_blocks = []
        for i in range(n_res_block):
            res_blocks.append(ResidualBlock(num_filters[-1], num_filters[-1]))

        self.res_blocks = nn.Sequential(*res_blocks)

        self.decoders = Decoder(in_channel=num_filters[-1], num_filters=list(reversed(num_filters)))

        self.img_reg = nn.Sequential(
            nn.Conv2d(num_filters[0], 3, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

        self.att_reg = nn.Sequential(
            nn.Conv2d(num_filters[0], 1, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc_x = self.encoders(x, get_details=False)
        # print("enc = {}".format(enc_x.shape))

        res_x = self.res_blocks(enc_x)
        # print("rex = {}".format(res_x.shape)

        dec_x = self.decoders(res_x)
        # print("dec = {}".format(dec_x.shape))
        return self.regress(dec_x)

    def decode(self, x):
        return self.decoders(x)

    def regress(self, x):
        return self.img_reg(x), self.att_reg(x)

    def encode(self, x):
        return self.encoders(x)

    def res_out(self, x):
        res_outs = []
        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)
            res_outs.append(x)

        return res_outs


class BaseLWBGenerator(nn.Module, ABC):
    def __init__(
        self, cfg, temporal=True
    ):

        super(BaseLWBGenerator, self).__init__()
        self.bg_net = ResNetInpaintor(
            c_dim=cfg.BGNet.cond_nc,
            num_filters=cfg.BGNet.num_filters,
            n_res_block=cfg.BGNet.n_res_block
        )

        # build src_net
        self.src_net = ResAutoEncoder(
            in_channel=cfg.SIDNet.cond_nc,
            num_filters=cfg.SIDNet.num_filters,
            n_res_block=cfg.SIDNet.n_res_block
        )

        # build tsf_net
        num_filters = cfg.TSFNet.num_filters
        n_res_block = cfg.TSFNet.n_res_block
        self.temporal = temporal
        self.tsf_net_enc = Encoder(
            in_channel=cfg.TSFNet.cond_nc,
            num_filters=num_filters,
            use_bias=False
        )
        self.tsf_net_dec = SkipDecoder(
            num_filters[-1],
            num_filters,
            list(reversed(num_filters))
        )

        res_blocks = []
        for i in range(n_res_block):
            res_blocks.append(ResidualBlock(num_filters[-1], num_filters[-1]))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.tsf_img_reg = nn.Sequential(
            nn.Conv2d(num_filters[0], 3, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

        self.tsf_att_reg = nn.Sequential(
            nn.Conv2d(num_filters[0], 1, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Sigmoid()
        )

        # child class must define self.lwb
        self.lwb = None

    def forward_bg(self, bg_inputs):
        """
        Args:
            bg_inputs (torch.tensor): (bs, ns, 4, h, w)

        Returns:
            bg_img (torch.tensor): the `viewed` bg_img from (bs * ns, 3, h, w) to (bs, ns, 3, h, w)
        """

        bs, ns, _, h, w = bg_inputs.shape

        bg_img = self.bg_net(bg_inputs.view(bs * ns, -1, h, w))
        bg_img = bg_img.view(bs, ns, 3, h, w)

        return bg_img

    def forward_src(self, src_inputs, only_enc=True):
        """

        Args:
            src_inputs (torch.tensor): (bs, ns, 6, h, w)
            only_enc (bool): the flag to control only encode or return the all outputs, including,
                encoder outputs, predicted img and mask map.

        Returns:
            enc_outs (list of torch.tensor): [torch.tensor(bs*ns, c1, h1, w1), tensor.tensor(bs*ns, c2, h2, w2), ... ]
            img (torch.tensor): if `only_enc == True`, return the predicted image map (bs, ns, 3, h, w).
            mask (torch.tensor): if `only_enc == True`, return the predicted mask map (bs, ns, 3, h, w)
        """
        bs, ns, _, h, w = src_inputs.shape
        src_enc_outs = self.src_net.encode(src_inputs.view(bs * ns, -1, h, w))
        src_res_outs = self.src_net.res_out(src_enc_outs[-1])

        if only_enc:
            return src_enc_outs, src_res_outs
        else:
            img, mask = self.src_net.regress(self.src_net.decode(src_res_outs[-1]))
            img = img.view(bs, ns, 3, h, w)
            mask = mask.view(bs, ns, 1, h, w)

            return src_enc_outs, src_res_outs, img, mask

    def forward_tsf(self, tsf_inputs, src_enc_outs, src_res_outs, Tst,
                    temp_enc_outs=None, temp_res_outs=None, Ttt=None):
        """
            Processing one time step of tsf stream.

        Args:
            tsf_inputs (torch.tensor): (bs, 6, h, w)
            src_enc_outs (list of torch.tensor): [(bs*ns, c1, h1, w1), (bs*ns, c2, h2, w2),..]
            src_res_outs (list of torch.tensor): [(bs*ns, c1, h1, w1), (bs*ns, c2, h2, w2),..]
            Tst (torch.tensor): (bs, ns, h, w, 2), flow transformation from source images/features
            temp_enc_outs (list of torch.tensor): [(bs*nt, c1, h1, w1), (bs*nt, c2, h2, w2),..]
            Ttt (torch.tensor): (bs, nt, h, w, 2), flow transformation from previous images/features (temporal smooth)

        Returns:
            tsf_enc_outs (list of torch.tensor):
            tsf_img (torch.tensor):  (bs, 3, h, w)
            tsf_mask (torch.tensor): (bs, 1, h, w)
        """
        bs, ns, h, w, _ = Tst.shape

        n_down = self.tsf_net_enc.n_down

        # 1. encoders
        tsf_enc_outs = []
        tsf_x = tsf_inputs
        Tst = Tst.view((bs * ns, h, w, 2))
        for i in range(n_down):
            tsf_x = self.tsf_net_enc.layers[i](tsf_x)
            src_x = src_enc_outs[i]

            tsf_x = self.lwb(tsf_x, src_x, Tst)

            tsf_enc_outs.append(tsf_x)

        # 2. res-blocks
        for i in range(len(self.res_blocks)):
            tsf_x = self.res_blocks[i](tsf_x)
            src_x = src_res_outs[i]

            tsf_x = self.lwb(tsf_x, src_x, Tst)

        # 3. decoders
        tsf_x = self.tsf_net_dec(tsf_x, tsf_enc_outs)
        tsf_img, tsf_mask = self.tsf_img_reg(tsf_x), self.tsf_att_reg(tsf_x)

        return tsf_img, tsf_mask

    def forward(self, bg_inputs, src_inputs, tsf_inputs, Tst, Ttt=None, only_tsf=True):
        """

        Args:
            bg_inputs (torch.tensor):   (bs, ns, 4, H, W)
            src_inputs (torch.tensor):  (bs, ns, 6, H, W)
            tsf_inputs (torch.tensor):  (bs, nt, 3 or 6, H, W)
            Tst (torch.tensor):         (bs, nt, ns, H, W, 2)
            Ttt (torch.tensor or None): (bs, nt - 1, H, H, 2)
            only_tsf (bool):

        Returns:
            bg_img (torch.tensor): the inpainted bg images, (bs, ns or 1, 3, h, w)
        """

        # print(src_inputs.shape, Tst.shape, Ttt.shape)
        bs, nt, ns, h, w, _ = Tst.shape

        # 1. inpaint background
        bg_img = self.forward_bg(bg_inputs)    # (N, ns or 1, 3, h, w)

        # 2. process source inputs
        # src_enc_outs: [torch.tensor(bs*ns, c1, h1, w1), tensor.tensor(bs*ns, c2, h2, w2), ... ]
        # src_img: the predicted image map (bs, ns, 3, h, w)
        # src_mask: the predicted mask map (bs, ns, 3, h, w)

        if only_tsf:
            src_enc_outs, src_res_outs = self.forward_src(src_inputs, only_enc=True)
            src_imgs, src_masks = None, None
        else:
            src_enc_outs, src_res_outs, src_imgs, src_masks = self.forward_src(src_inputs, only_enc=False)

        # 3. process transform inputs
        tsf_imgs, tsf_masks = [], []
        for t in range(nt):
            t_tsf_inputs = tsf_inputs[:, t]

            if t != 0 and self.temporal:
                _tsf_cond = tsf_inputs[:, t - 1, 0:3]
                _tsf_img = tsf_imgs[-1] * (1 - tsf_masks[-1])
                _tsf_inputs = torch.cat([_tsf_img, _tsf_cond], dim=1).unsqueeze_(dim=1)
                _temp_enc_outs, _temp_res_outs = self.forward_src(_tsf_inputs, only_enc=True)
                _Ttt = Ttt[:, t-1:t]
            else:
                _Ttt = None
                _temp_enc_outs, _temp_res_outs = None, None
            tsf_img, tsf_mask = self.forward_tsf(t_tsf_inputs, src_enc_outs, src_res_outs,
                                                 Tst[:, t].contiguous(), _temp_enc_outs, _temp_res_outs, _Ttt)
            tsf_imgs.append(tsf_img)
            tsf_masks.append(tsf_mask)

        tsf_imgs = torch.stack(tsf_imgs, dim=1)
        tsf_masks = torch.stack(tsf_masks, dim=1)

        if only_tsf:
            return bg_img, tsf_imgs, tsf_masks
        else:
            return bg_img, src_imgs, src_masks, tsf_imgs, tsf_masks


class AddLWBGenerator(BaseLWBGenerator):
    def __init__(
        self, cfg, temporal=True
    ):
        super(AddLWBGenerator, self).__init__(cfg, temporal)

        self.lwb = AddLWB()


class AvgLWBGenerator(BaseLWBGenerator):
    def __init__(
        self, cfg, temporal=True
    ):
        super(AvgLWBGenerator, self).__init__(cfg, temporal)

        self.lwb = AvgLWB()
