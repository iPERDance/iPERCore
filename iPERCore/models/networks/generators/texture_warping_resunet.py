import torch
import torch.nn as nn

from .bg_inpaintor import ResNetInpaintor
from .input_concat_resunet import ResAutoEncoder


class TextureWarpingGenerator(nn.Module):
    def __init__(self, cfg, temporal=False):
        super(TextureWarpingGenerator, self).__init__()

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

        tsf_img, tsf_mask = self.tsf_net.forward(tsf_inputs)

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

        bs, nt, c, h, w = tsf_inputs.shape

        tsf_inputs = tsf_inputs.view(bs * nt, -1, h, w)
        tsf_imgs, tsf_masks = self.tsf_net.forward(tsf_inputs)

        tsf_imgs = tsf_imgs.view(bs, nt, 3, h, w)
        tsf_masks = tsf_masks.view(bs, nt, 1, h, w)

        return bg_img, tsf_imgs, tsf_masks

