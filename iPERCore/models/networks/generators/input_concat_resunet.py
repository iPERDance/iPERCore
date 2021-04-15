import torch
import torch.nn as nn

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


class InputConcatGenerator(nn.Module):
    def __init__(self, cfg, temporal=False):
        super(InputConcatGenerator, self).__init__()

        self.bg_net = ResNetInpaintor(
            c_dim=cfg.BGNet.cond_nc,
            num_filters=cfg.BGNet.num_filters,
            n_res_block=cfg.BGNet.n_res_block
        )
        self.tsf_net = ResAutoEncoder(
            in_channel=cfg.TSFNet.cond_nc,
            num_filters=cfg.TSFNet.num_filters,
            n_res_block=cfg.TSFNet.n_res_block
        )

        self.num_source = cfg.TSFNet.num_source

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
        need_ns = self.num_source

        if ns > need_ns:
            src_inputs = src_inputs[:, 0:need_ns]
        elif ns < need_ns:
            need_pad_ns = need_ns - ns

            pad_src_inputs = []
            for s in range(need_pad_ns):
                pad_src_inputs.append(src_inputs[:, s % ns])

            pad_src_inputs = torch.stack(pad_src_inputs, dim=1)
            src_inputs = torch.cat([src_inputs, pad_src_inputs], dim=1)

        src_enc_outs = src_inputs.view(bs, -1, h, w)

        if only_enc:
            return src_enc_outs, src_enc_outs
        else:
            return src_enc_outs, src_enc_outs, None, None

    def forward_tsf(self, tsf_inputs, src_enc_outs, src_res_outs=None, Tst=None,
                    temp_enc_outs=None, temp_res_outs=None, Ttt=None):
        """
            Processing one time step of tsf stream.

        Args:
            tsf_inputs (torch.tensor): (bs, 6, h, w)
            src_enc_outs (torch.tensor): (bs, c, h, wc)
            src_res_outs (None): None
            Tst (torch.tensor): (bs, ns, h, w, 2), flow transformation from source images/features
            temp_enc_outs (list of torch.tensor): [(bs*nt, c1, h1, w1), (bs*nt, c2, h2, w2),..]
            temp_res_outs (list of torch.tensor): [(bs*nt, c1, h1, w1), (bs*nt, c2, h2, w2),..]
            Ttt (torch.tensor): (bs, nt, h, w, 2), flow transformation from previous images/features (temporal smooth)

        Returns:
            tsf_enc_outs (list of torch.tensor):
            tsf_img (torch.tensor):  (bs, 3, h, w)
            tsf_mask (torch.tensor): (bs, 1, h, w)
        """

        tsf_cond = tsf_inputs[:, -3:]
        inputs = torch.cat([src_enc_outs, tsf_cond], dim=1)

        tsf_img, tsf_mask = self.tsf_net.forward(inputs)

        return tsf_img, tsf_mask

    def forward(self, bg_inputs, src_inputs, tsf_inputs, Tst=None, Ttt=None, only_tsf=True):
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

        # 1. inpaint background
        bg_img = self.forward_bg(bg_inputs)  # (N, ns or 1, 3, h, w)

        src_enc_outs, _ = self.forward_src(src_inputs, only_enc=True)

        bs, nt = tsf_inputs.shape[0:2]
        tsf_imgs, tsf_masks = [], []
        for t in range(nt):
            tsf_img, tsf_mask = self.forward_tsf(tsf_inputs[:, t], src_enc_outs)
            tsf_imgs.append(tsf_img)
            tsf_masks.append(tsf_mask)

        tsf_imgs = torch.stack(tsf_imgs, dim=1)
        tsf_masks = torch.stack(tsf_masks, dim=1)

        return bg_img, tsf_imgs, tsf_masks


class TextureWarpingGenerator(InputConcatGenerator):
    def __init__(self, cfg, temporal=False):
        super().__init__(cfg, temporal)

