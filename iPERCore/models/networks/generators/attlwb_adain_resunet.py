# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

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


class AdaIN(nn.Module):
    def __init__(self, style_fact=1.0, eps=1e-5):
        super(AdaIN, self).__init__()
        self.style_fact = style_fact
        self.eps = eps

    def forward(self, content, gamma, beta):
        """
        Args:
            content (torch.tensor): (b, c, h, w)
            gamma (torch.tensor):   (b, c1, h, w)
            beta  (torch.tensor):   (b, c2, h, w)

        Returns:

        """
        b, c, h, w = content.size()

        # (b, 1, h, w)
        content_std, content_mean = torch.std_mean(content, dim=1, keepdim=True)

        normalized_content = (content - content_mean) / (content_std + self.eps)

        stylized_content = (normalized_content * gamma) + beta

        if self.style_fact != 1.0:
            output = (1 - self.style_fact) * content + self.style_fact * stylized_content
        else:
            output = stylized_content
        return output


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
        x = torch.sum(alpha * v, dim=1)     # (N, ns, C, H, W) * (N, ns, 1, 1, 1)

        return x

    def query(self, q, k):
        """

        Args:
            q (torch.tensor): (N, C, H, W)
            k (torch.tensor): (N, ns, C, H, W)

        Returns:
            alpha (torch.tensor): (N, ns, 1, H, W)
        """

        bs, ns, c, h, w = k.shape
        dk = c * h * w

        q = torch.mean(q, dim=(2, 3))
        k = torch.mean(k, dim=(3, 4))

        alpha = torch.matmul(k.view(bs, ns, -1), q.view(bs, -1).unsqueeze(-1)) / math.sqrt(dk)    # (N, ns, 1))
        alpha = torch.softmax(alpha, dim=1)             # (N, ns, 1)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)

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
        T_scale = F.interpolate(T_scale, size=(h, w), mode='bilinear', align_corners=True)
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

        gamma, beta = torch.std_mean(x, dim=1, keepdim=True)

        return x, gamma, beta


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
        self._name = 'ResAutoEncoder'

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


class AttentionLWBGenerator(nn.Module):
    def __init__(
        self, bg_dim=4, src_dim=6, tsf_dim=3,
        num_filters=(64, 128, 128, 128), n_res_block=4,
        temporal=True

    ):
        super(AttentionLWBGenerator, self).__init__()
        self.bg_net = ResNetInpaintor(c_dim=bg_dim, num_filters=num_filters, n_res_block=6)

        # build src_net
        num_filters = (64, 128, 256)
        self.src_net = ResAutoEncoder(in_channel=src_dim, num_filters=num_filters, n_res_block=n_res_block)

        # build tsf_net
        self.temporal = temporal
        self.tsf_net_enc = Encoder(in_channel=tsf_dim, num_filters=num_filters, use_bias=False)
        self.tsf_net_dec = SkipDecoder(num_filters[-1], num_filters, list(reversed(num_filters)))
        self.enc_attlwbs = build_multi_stage_attlwb(num_filters, num_filters, temporal)
        self.res_attlwbs = build_res_block_attlwb(num_filters[-1], num_filters[-1], n_res_block, temporal)
        res_blocks = []
        for i in range(n_res_block):
            res_blocks.append(ResidualBlock(num_filters[-1], num_filters[-1]))
        self.res_blocks = nn.Sequential(*res_blocks)
        self.adain = AdaIN()

        self.tsf_img_reg = nn.Sequential(
            nn.Conv2d(num_filters[0], 3, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

        self.tsf_att_reg = nn.Sequential(
            nn.Conv2d(num_filters[0], 1, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Sigmoid()
        )

    def adaptor_params(self):
        return list(self.bg_net.parameters()) +      \
               list(self.tsf_net_enc.parameters()) + \
               list(self.tsf_net_dec.parameters()) + \
               list(self.res_blocks.parameters()) +  \
               list(self.enc_attlwbs.parameters()) + list(self.res_attlwbs.parameters()) + \
               list(self.tsf_img_reg.parameters()) + list(self.tsf_att_reg.parameters())

    def adaptor_att_params(self):
        return list(self.enc_attlwbs.parameters()) + list(self.res_attlwbs.parameters())

    def adaptor_params_for_swapper(self):
        return list(self.bg_net.parameters()) + \
               list(self.src_net.att_reg.parameters()) + \
               list(self.enc_attlwbs.parameters()) + list(self.res_attlwbs.parameters())

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
        for i in range(n_down):
            tsf_x = self.tsf_net_enc.layers[i](tsf_x)
            src_x = src_enc_outs[i]

            if temp_enc_outs is not None and Ttt is not None:
                temp_x = temp_enc_outs[i]
            else:
                temp_x = None

            # print(q_x.shape, src_x.shape, Tst.shape)
            x, gamma, beta = self.enc_attlwbs[i](tsf_x, src_x, Tst, temp_x=temp_x, Ttt=Ttt)

            tsf_x = self.adain(tsf_x, gamma, beta)
            tsf_enc_outs.append(tsf_x)

        # 2. res-blocks
        for i in range(len(self.res_blocks)):
            tsf_x = self.res_blocks[i](tsf_x)
            src_x = src_res_outs[i]
            if temp_enc_outs is not None and Ttt is not None:
                temp_x = temp_res_outs[i]
            else:
                temp_x = None
            x, gamma, beta = self.res_attlwbs[i](tsf_x, src_x, Tst, temp_x=temp_x, Ttt=Ttt)
            tsf_x = self.adain(tsf_x, gamma, beta)

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


if __name__ == '__main__':
    alwb_gen = AttentionLWBGenerator(temporal=True, num_filters=3)

    bg_inputs = torch.rand(4, 5, 4, 512, 512)
    src_inputs = torch.rand(4, 5, 6, 512, 512)
    tsf_inputs = torch.rand(4, 2, 3, 512, 512)
    Tst = torch.rand(4, 2, 5, 512, 512, 2)
    Ttt = torch.rand(4, 1, 512, 512, 2)

    bg_img, src_img, src_mask, tsf_img, tsf_mask = alwb_gen(bg_inputs, src_inputs, tsf_inputs, Tst, Ttt)

    print(bg_img.shape, src_img.shape, src_mask.shape, tsf_img.shape, tsf_mask.shape)

