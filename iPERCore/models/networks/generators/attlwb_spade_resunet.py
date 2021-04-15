# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import re
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


def calc_std_mean(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_std, feat_mean


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, cond_nc):
        super().__init__()

        assert config_text.startswith("spade")
        parsed = re.search(r"spade(\D+)(\d)x\d", config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError("%s is not a recognized param-free norm type in SPADE"
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, condmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # condmap = F.interpolate(condmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(condmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out

    def cal_gamma_beta(self, condmap):
        actv = self.mlp_shared(condmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return gamma, beta


class SelfAttentionBlock(nn.Module):
    def __init__(self):
        super(SelfAttentionBlock, self).__init__()

    def forward(self, q, k, v):
        """
        Args:
            q (torch.tensor): (N, C, H, W)
            k (torch.tensor): (N, ns, C, H, W)
            v (torch.tensor): (N, ns, C, H, W)
        Returns:
            x (torch.tensor): (N, C, H, W)
        """

        alpha = self.query(q, k)            # (N, ns, 1, H, W)
        x = torch.sum(alpha * v, dim=1)     # (N, ns, C, H, W) * (N, ns, 1, H, W)

        return x

    def query(self, q, k):
        """

        Args:
            q (torch.tensor): (N, C, H, W)
            k (torch.tensor): (N, ns, C, H, W)

        Returns:
            alpha (torch.tensor): (N, ns, 1, H, W)
        """
        dk = k.shape[2]
        q = q.permute(0, 2, 3, 1)        # (N, C, H, W) - > (N, H, W, C)
        k = k.permute(0, 3, 4, 1, 2)     # (N, ns, C, H, W) -> (N, H, W, ns, C)
        q.unsqueeze_(-1)
        alpha = torch.matmul(k, q) / math.sqrt(dk)      # (N, H, W, ns, 1)
        alpha = torch.softmax(alpha, dim=-2)            # (N, H, W, ns, 1)
        alpha = alpha.permute(0, 3, 4, 1, 2)            # (N, ns, 1, H, W)

        return alpha


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


class SelfAttentionLWB(nn.Module):
    def __init__(self, channel_q, channel_s, channel, temporal=False):
        super().__init__()

        self.temporal = temporal

        self.att_block = SelfAttentionBlock()
        self.lwb = LWB()
        self.fq = nn.Conv2d(channel_q, channel, kernel_size=1)
        self.fk = nn.Conv2d(channel_s, channel, kernel_size=1)
        self.fv = nn.Conv2d(channel_s, channel, kernel_size=1)

        self.spade = SPADE("spadeinstance3x3", channel_q, channel)

    def forward(self, tsf_x, src_x, Tst, temp_x=None, Ttt=None):
        """
        Args:
            tsf_x (torch.tensor):  (bs, c1, h, w)
            src_x (torch.tensor):  (bs * ns, c2, h, w)
            Tst (torch.tensor):    (bs, ns, h, w, 2)
            temp_x (torch.tensor or None): (bs * nt, c, H, W)
            Ttt (torch.tensor or None):    (bs, nt, H, W, 2)

        Returns:
            x (torch.tensor):     (bs, c, h, w)
            gamma (torch.tensor): (bs, 3, h, w)
            beta (torch.tensor) : (bs, 3, h, w)
        """
        bs, ns, H, W, _ = Tst.shape
        h, w = tsf_x.shape[-2:]

        src_warp = self.lwb(src_x, Tst.view(bs * ns, H, W, 2))         # (bs * ns, c2, h, w)
        src_warp_k = self.fk(src_warp)         # (bs * ns, c, h, w)
        src_warp_v = self.fv(src_warp)         # (bs * ns, c, h, w)

        K = [src_warp_k.view(bs, ns, -1, h, w)] # (bs, ns, c, h, w)
        V = [src_warp_v.view(bs, ns, -1, h, w)] # (bs, ns, c, h, w)

        if self.temporal and temp_x is not None and Ttt is not None:
            nt = Ttt.shape[1]
            # print(temp_x.shape, Ttt.shape)
            temp_warp = self.lwb(temp_x, Ttt.view(bs * nt, H, W, 2))   # (bs * nt, c, h, w)
            temp_warp_k = self.fk(temp_warp)   # (bs * nt, c, h, w)
            temp_warp_v = self.fv(temp_warp)   # (bs * nt, c, h, w)

            K.append(temp_warp_k.view(bs, nt, -1, h, w))   # (bs, nt, c, h, w)
            V.append(temp_warp_v.view(bs, nt, -1, h, w))   # (bs, nt, c, h, w)

        K = torch.cat(K, dim=1)     # (bs, ns + nt, c, h, w)
        V = torch.cat(V, dim=1)     # (bs, ns + nt, c, h, w)

        q = self.fq(tsf_x)  # (bs, c, h, w)

        # attention query
        x = self.att_block(q, K, V)  # (bs, c, h, w)

        tsf_x_normalized = self.spade(tsf_x, x)

        return tsf_x_normalized


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
        self.encoders = Encoder(in_channel=in_channel, num_filters=num_filters, use_bias=True)

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


def build_multi_stage_attlwb(src_num_filters, tsf_num_filters, temporal=True):
    assert len(src_num_filters) == len(tsf_num_filters)

    enc_attlwbs = nn.ModuleList()
    # Down-Sampling
    n_down = len(src_num_filters)
    for i in range(n_down):
        block = SelfAttentionLWB(
            channel_q=tsf_num_filters[i],
            channel_s=src_num_filters[i],
            channel=tsf_num_filters[i],
            temporal=temporal
        )

        enc_attlwbs.append(block)
    return enc_attlwbs


def build_res_block_attlwb(src_res_channel, tsf_res_channel, n_res_block, temporal=True):
    res_attlwbs = nn.ModuleList()
    # Down-Sampling
    for i in range(n_res_block):
        block = SelfAttentionLWB(
            channel_q=tsf_res_channel,
            channel_s=src_res_channel,
            channel=tsf_res_channel,
            temporal=temporal
        )

        res_attlwbs.append(block)
    return res_attlwbs


class BaseAttentionLWBGenerator(nn.Module, ABC):

    def forward_src(self, src_inputs, only_enc=True):
        """
        Run forward of SIDNet, and get the reconstructed source images.
        If only_enc is True, then it only returns the features at each stage of the source images;
        Otherwise, it returns the features, and the recontructed source images, as well as the masks.

        Args:
            src_inputs (torch.tensor): (bs, ns, 6, h, w)
            only_enc (bool): the flag to control only encode or return the all outputs, including,
                encoder outputs, predicted img and mask map.

        Returns:
            src_enc_outs (list of torch.Tensor): [torch.tensor(bs*ns, c1, h1, w1), tensor.tensor(bs*ns, c2, h2, w2), ...];
            src_res_outs (list of torch.Tensor): [torch.tensor(bs*ns, ck, hk, wk), tensor.tensor(bs*ns, ck, hk, wk), ...];
            img (torch.Tensor): if `only_enc == True`, return the predicted image map (bs, ns, 3, h, w);
            mask (torch.Tensor): if `only_enc == True`, return the predicted mask map (bs, ns, 3, h, w).
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
            Processing one time step of TSFNet.

        Args:
            tsf_inputs (torch.Tensor): (bs, 6, h, w)
            src_enc_outs (list of torch.Tensor): [(bs*ns, c1, h1, w1), (bs*ns, c2, h2, w2),..]
            src_res_outs (list of torch.Tensor): [(bs*ns, c1, h1, w1), (bs*ns, c2, h2, w2),..]
            Tst (torch.Tensor): (bs, ns, h, w, 2), flow transformation from source images/features
            temp_enc_outs (list of torch.Tensor): [(bs*nt, c1, h1, w1), (bs*nt, c2, h2, w2),..]
            temp_res_outs (list of torch.Tensor): [(bs*nt, ck, hk, wk), (bs*nt, ck, hk, wk),..]
            Ttt (torch.Tensor): (bs, nt, h, w, 2), flow transformation from previous images/features (temporal smooth)

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
        for i in range(n_down):
            tsf_x = self.tsf_net_enc.layers[i](tsf_x)
            src_x = src_enc_outs[i]

            if temp_enc_outs is not None and Ttt is not None:
                temp_x = temp_enc_outs[i]
            else:
                temp_x = None

            # print(q_x.shape, src_x.shape, Tst.shape)
            tsf_x = self.enc_attlwbs[i](tsf_x, src_x, Tst, temp_x=temp_x, Ttt=Ttt)

            tsf_enc_outs.append(tsf_x)

        # 2. res-blocks
        for i in range(len(self.res_blocks)):
            tsf_x = self.res_blocks[i](tsf_x)
            src_x = src_res_outs[i]
            if temp_enc_outs is not None and Ttt is not None:
                temp_x = temp_res_outs[i]
            else:
                temp_x = None
            tsf_x = self.res_attlwbs[i](tsf_x, src_x, Tst, temp_x=temp_x, Ttt=Ttt)

        # 3. decoders
        tsf_x = self.tsf_net_dec(tsf_x, tsf_enc_outs)
        tsf_img, tsf_mask = self.tsf_img_reg(tsf_x), self.tsf_att_reg(tsf_x)

        return tsf_img, tsf_mask


class AttentionLWBGenerator(BaseAttentionLWBGenerator):

    def __init__(self, cfg, temporal=False):
        """

        Args:
            cfg (dict or EasyDict): the configurations, it contains the followings:
                --name (str): the name of Generator;
                --BGNet (dict or EasyDict): the configurations of BGNet, and it contains the followings:
                    --norm_type (str): the type of normalization, default is "instance";
                    --cond_nc (int):  the number channels of conditions, default is RGB (3) + MASK (1) = 4;
                    --n_res_block (int): the number of residual blocks, default is 6;
                    --num_filters (list): the number of filters, default is [64, 128, 128, 256].

                --SIDNet (dict or EasyDict): the configurations of SIDNet, and it contains the followings:
                    --norm_type (str): the type of normalization, default is "None", do not use any normalization;
                    --cond_nc (int): the number of conditions, default is RGB (3) + UV_Seg (3) = 6;
                    --n_res_block (int): the number of residual blocks, default is 6;
                    --num_filters (list): the number of filters, default is [64, 128, 256].

                --TSFNet (dict or EasyDict): the configurations of TSFNet, and it contains the followings:
                    --norm_type (str): the type of normalization, default is "instance";
                    --cond_nc (int): the number of conditions, default is RGB (3) + UV_Seg (3) = 6;
                    --n_res_block (int): the number of residual blocks, default is 6;
                    --num_filters (list): the number of filters, default is [64, 128, 256].

            temporal (bool): use temporal attention or not, default is False.
        """

        super(AttentionLWBGenerator, self).__init__()
        self._name = cfg.name

        # 1. build BGNet.
        self.bg_net = ResNetInpaintor(
            c_dim=cfg.BGNet.cond_nc,
            num_filters=cfg.BGNet.num_filters,
            n_res_block=cfg.BGNet.n_res_block
        )

        # 2. build SIDNet
        self.src_net = ResAutoEncoder(
            in_channel=cfg.SIDNet.cond_nc,
            num_filters=cfg.SIDNet.num_filters,
            n_res_block=cfg.SIDNet.n_res_block
        )

        # 3. build TSFNet
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
        self.enc_attlwbs = build_multi_stage_attlwb(num_filters, num_filters, temporal)
        self.res_attlwbs = build_res_block_attlwb(num_filters[-1], num_filters[-1], n_res_block, temporal)
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

    def forward_bg(self, bg_inputs):
        """
        Run forward of BGNet, and predict the inpainted results.

        Args:
            bg_inputs (torch.tensor): (bs, ns, 4, h, w)

        Returns:
            bg_img (torch.tensor): the `viewed` bg_img from (bs * ns, 3, h, w) to (bs, ns, 3, h, w)
        """

        bs, ns, _, h, w = bg_inputs.shape

        bg_img = self.bg_net(bg_inputs.view(bs * ns, -1, h, w))
        bg_img = bg_img.view(bs, ns, 3, h, w)

        return bg_img

    def forward(self, bg_inputs, src_inputs, tsf_inputs, Tst, Ttt=None, only_tsf=True):
        """
        Forward to get all results.
        If only_tsf is True, then it returns [bg_img, src_imgs, src_masks, tsf_imgs, tsf_masks];
        Otherwise, then it only returns [bg_img, tsf_imgs, tsf_masks].

        Args:
            bg_inputs (torch.tensor):   (bs, ns, 4, H, W)
            src_inputs (torch.tensor):  (bs, ns, 6, H, W)
            tsf_inputs (torch.tensor):  (bs, nt, 3 or 6, H, W)
            Tst (torch.tensor):         (bs, nt, ns, H, W, 2)
            Ttt (torch.tensor or None): (bs, nt - 1, H, H, 2)
            only_tsf (bool): whether only returns the TSFNet or not.

        Returns:
            bg_img (torch.Tensor): the inpainted bg images, (bs, ns or 1, 3, h, w);
            src_imgs (torch.Tensor): the reconstructed source images, (bs, ns, 3, h, w);
            src_masks (torch.Tensor): the reconstructed source masks, (bs, ns, 1, h, w):
            tsf_imgs (torch.Tensor): the synthesized images, (bs, nt, 3, h, w):
            tsf_masks (torch.Tensor): the synthesized masks, (bs, nt, 1, h, w).

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


class AttentionLWBFrontGenerator(BaseAttentionLWBGenerator):

    def __init__(self, cfg, temporal=False):
        """

        Args:
            cfg (dict or EasyDict): the configurations, it contains the followings:
                --name (str): the name of Generator;
                --BGNet (dict or EasyDict): the configurations of BGNet, and it contains the followings:
                    --norm_type (str): the type of normalization, default is "instance";
                    --cond_nc (int):  the number channels of conditions, default is RGB (3) + MASK (1) = 4;
                    --n_res_block (int): the number of residual blocks, default is 6;
                    --num_filters (list): the number of filters, default is [64, 128, 128, 256].

                --SIDNet (dict or EasyDict): the configurations of SIDNet, and it contains the followings:
                    --norm_type (str): the type of normalization, default is "None", do not use any normalization;
                    --cond_nc (int): the number of conditions, default is RGB (3) + UV_Seg (3) = 6;
                    --n_res_block (int): the number of residual blocks, default is 6;
                    --num_filters (list): the number of filters, default is [64, 128, 256].

                --TSFNet (dict or EasyDict): the configurations of TSFNet, and it contains the followings:
                    --norm_type (str): the type of normalization, default is "instance";
                    --cond_nc (int): the number of conditions, default is RGB (3) + UV_Seg (3) = 6;
                    --n_res_block (int): the number of residual blocks, default is 6;
                    --num_filters (list): the number of filters, default is [64, 128, 256].

            temporal (bool): use temporal attention or not, default is False.
        """

        super(AttentionLWBFrontGenerator, self).__init__()

        self._name = cfg.name

        # 1. build SIDNet
        self.src_net = ResAutoEncoder(
            in_channel=cfg.SIDNet.cond_nc,
            num_filters=cfg.SIDNet.num_filters,
            n_res_block=cfg.SIDNet.n_res_block
        )

        # 2. build TSFNet
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
        self.enc_attlwbs = build_multi_stage_attlwb(num_filters, num_filters, temporal)
        self.res_attlwbs = build_res_block_attlwb(num_filters[-1], num_filters[-1], n_res_block, temporal)
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

    def forward(self, src_inputs, tsf_inputs, Tst, Ttt=None, only_tsf=True):
        """
        Forward to get all results.
        If only_tsf is True, then it returns [src_imgs, src_masks, tsf_imgs, tsf_masks];
        Otherwise, then it only returns [tsf_imgs, tsf_masks].

        Args:
            src_inputs (torch.tensor):  (bs, ns, 6, H, W)
            tsf_inputs (torch.tensor):  (bs, nt, 3 or 6, H, W)
            Tst (torch.tensor):         (bs, nt, ns, H, W, 2)
            Ttt (torch.tensor or None): (bs, nt - 1, H, H, 2)
            only_tsf (bool): whether only returns the TSFNet or not.

        Returns:
            src_imgs (torch.Tensor): the reconstructed source images, (bs, ns, 3, h, w);
            src_masks (torch.Tensor): the reconstructed source masks, (bs, ns, 1, h, w):
            tsf_imgs (torch.Tensor): the synthesized images, (bs, nt, 3, h, w):
            tsf_masks (torch.Tensor): the synthesized masks, (bs, nt, 1, h, w).

        """

        # print(src_inputs.shape, Tst.shape, Ttt.shape)

        bs, nt, ns, h, w, _ = Tst.shape

        # 1. process source inputs
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
            return tsf_imgs, tsf_masks
        else:
            return src_imgs, src_masks, tsf_imgs, tsf_masks


