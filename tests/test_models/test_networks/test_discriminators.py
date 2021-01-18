# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import unittest
import torch
import toml
from easydict import EasyDict

from iPERCore.models.networks import discriminators


class TestGenerators(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.patch_dis_cfg_str = """
            [Generator]
            # the configuration of the generator, since the generator contains the 3 sub-networks, they are
            # BGNet, SIDNet, TSFNet

            name = "AttLWB-SPADE"

                [Generator.BGNet]
                    # the configuration of the Background Inpainting Network
                    norm_type = "instance"             # use instance normalization
                    cond_nc = 4                        # number channels of conditions, RGB (3) + MASK (1) = 4
                    n_res_block = 6                    # number of residual blocks
                    num_filters = [64, 128, 128, 256]  # number of filters

                [Generator.SIDNet]
                    # the configuration of the Source Identiy Network
                    norm_type = "None"                # do not use normalization;
                    cond_nc = 6                       # number of conditions, RGB (3) + UV_Seg (3) = 6
                    n_res_block = 6                   # number of residual blocks.
                    num_filters = [64, 128, 256]      # number of filters

                [Generator.TSFNet]
                    # the configuration of the Transfer Network
                    norm_type = "instance"            # use instance normalization
                    cond_nc = 6                       # number of conditions, RGB (3) + UV_Seg (3) = 6
                    n_res_block = 6                   # number of residual blocks.
                    num_filters = [64, 128, 256]      # number of filters

            [Discriminator]
            name = "patch_global"
                # the configuration of the discriminator
                norm_type = "instance"            # use instance normalization;
                cond_nc = 6         # the number of conditions
                bg_cond_nc = 4      # the number of background conditions
                ndf = 64            # the number of the base filters for Discriminator;
                n_layers = 4        # the number of downsampling operations, convolution with stride = 2;
                max_nf_mult = 8     # the max multi fact, [ndf, ndf * 2, ndf * 4, ndf * 8, ndf * 8, ndf * 8, ...];
                use_sigmoid = false # here, we do not use Sigmoid, since we use LSGAN.
        """

    def test_01_PatchDiscriminator(self):

        cfg_str = self.patch_dis_cfg_str
        cfg = EasyDict(toml.loads(cfg_str))

        dis_cfg = cfg["Discriminator"]

        input_nc = dis_cfg.cond_nc
        ndf = dis_cfg.ndf
        n_layers = dis_cfg.n_layers
        max_nf_mult = dis_cfg.max_nf_mult
        norm_type = dis_cfg.norm_type
        use_sigmoid = dis_cfg.use_sigmoid

        patch_dis = discriminators.PatchDiscriminator(
            input_nc=input_nc, ndf=ndf, n_layers=n_layers,
            max_nf_mult=max_nf_mult, norm_type=norm_type, use_sigmoid=use_sigmoid
        )

        x = torch.rand(4, 6, 512, 512)

        outs = patch_dis(x)

        print(outs.shape)
        self.assertEqual(tuple(outs.shape), (4, 1, 30, 30))

    def test_02_GlobalDiscriminator(self):

        cfg_str = self.patch_dis_cfg_str
        cfg = EasyDict(toml.loads(cfg_str))

        dis_cfg = cfg["Discriminator"]

        patch_dis = discriminators.GlobalDiscriminator(dis_cfg)

        x = torch.rand(4, 6, 512, 512)

        inputs = {
            "x": x,
            "bg_x": None
        }

        outs = patch_dis(inputs)

        print(outs[0].shape)
        self.assertEqual(tuple(outs[0].shape), (4, 1, 30, 30))

    def test_03_GlobalLocalDiscriminator(self):

        cfg_str = self.patch_dis_cfg_str
        cfg = EasyDict(toml.loads(cfg_str))

        dis_cfg = cfg["Discriminator"]

        patch_dis = discriminators.GlobalLocalDiscriminator(dis_cfg)

        x = torch.rand(4, 6, 512, 512)
        body_rects = torch.zeros((4, 4), dtype=torch.long)
        body_rects[:, [0, 2]] = 50
        body_rects[:, [1, 3]] = 450

        inputs = {
            "x": x,
            "bg_x": None,
            "body_rects": body_rects,
            "get_avg": True
        }

        outs, avg = patch_dis(inputs)

        print(outs[0].shape, outs[1].shape, avg)

        self.assertEqual(len(outs), 2)
        self.assertEqual(tuple(outs[0].shape), (4, 1, 30, 30))
        self.assertEqual(tuple(outs[1].shape), (4, 1, 14, 14))

    def test_04_GlobalBodyHeadDiscriminator(self):

        cfg_str = self.patch_dis_cfg_str
        cfg = EasyDict(toml.loads(cfg_str))

        dis_cfg = cfg["Discriminator"]

        patch_dis = discriminators.GlobalBodyHeadDiscriminator(dis_cfg)

        x = torch.rand(4, 6, 512, 512)

        body_rects = torch.zeros((4, 4), dtype=torch.long)
        body_rects[:, [0, 2]] = 50
        body_rects[:, [1, 3]] = 450

        head_rects = torch.zeros((4, 4), dtype=torch.long)
        head_rects[:, [0, 2]] = 150
        head_rects[:, [1, 3]] = 250

        inputs = {
            "x": x,
            "bg_x": None,
            "body_rects": body_rects,
            "head_rects": head_rects,
            "get_avg": True
        }

        outs, avg = patch_dis(inputs)

        print(outs[0].shape, outs[1].shape, outs[2].shape, avg)

        self.assertEqual(len(outs), 3)
        self.assertEqual(tuple(outs[0].shape), (4, 1, 30, 30))
        self.assertEqual(tuple(outs[1].shape), (4, 1, 14, 14))
        self.assertEqual(tuple(outs[2].shape), (4, 1, 6, 6))

    def test_05_BGAugGlobalDiscriminator(self):

        cfg_str = self.patch_dis_cfg_str
        cfg = EasyDict(toml.loads(cfg_str))

        dis_cfg = cfg["Discriminator"]

        patch_dis = discriminators.GlobalDiscriminator(dis_cfg, use_aug_bg=True)

        x = torch.rand(4, 6, 512, 512)
        bg_x = torch.rand(4, 4, 512, 512)

        inputs = {
            "x": x,
            "bg_x": bg_x
        }

        outs = patch_dis(inputs)

        print(outs[0].shape, outs[1].shape)
        self.assertEqual(tuple(outs[0].shape), (4, 1, 30, 30))
        self.assertEqual(tuple(outs[1].shape), (4, 1, 30, 30))

    def test_06_BGAugGlobalLocalDiscriminator(self):

        cfg_str = self.patch_dis_cfg_str
        cfg = EasyDict(toml.loads(cfg_str))

        dis_cfg = cfg["Discriminator"]

        patch_dis = discriminators.GlobalLocalDiscriminator(dis_cfg, use_aug_bg=True)

        x = torch.rand(4, 6, 512, 512)
        bg_x = torch.rand(4, 4, 512, 512)
        body_rects = torch.zeros((4, 4), dtype=torch.long)
        body_rects[:, [0, 2]] = 50
        body_rects[:, [1, 3]] = 450

        inputs = {
            "x": x,
            "bg_x": bg_x,
            "body_rects": body_rects,
            "get_avg": True
        }

        outs, avg = patch_dis(inputs)

        print(outs[0].shape, outs[1].shape, outs[2].shape, avg)

        self.assertEqual(len(outs), 3)
        self.assertEqual(tuple(outs[0].shape), (4, 1, 30, 30))
        self.assertEqual(tuple(outs[1].shape), (4, 1, 30, 30))
        self.assertEqual(tuple(outs[2].shape), (4, 1, 14, 14))

    def test_07_BGAugGlobalBodyHeadDiscriminator(self):

        cfg_str = self.patch_dis_cfg_str
        cfg = EasyDict(toml.loads(cfg_str))

        dis_cfg = cfg["Discriminator"]

        patch_dis = discriminators.GlobalBodyHeadDiscriminator(dis_cfg, use_aug_bg=True)

        x = torch.rand(4, 6, 512, 512)
        bg_x = torch.rand(4, 4, 512, 512)

        body_rects = torch.zeros((4, 4), dtype=torch.long)
        body_rects[:, [0, 2]] = 50
        body_rects[:, [1, 3]] = 450

        head_rects = torch.zeros((4, 4), dtype=torch.long)
        head_rects[:, [0, 2]] = 150
        head_rects[:, [1, 3]] = 250

        inputs = {
            "x": x,
            "bg_x": bg_x,
            "body_rects": body_rects,
            "head_rects": head_rects,
            "get_avg": True
        }

        outs, avg = patch_dis(inputs)

        print(outs[0].shape, outs[1].shape, outs[2].shape, outs[3].shape, avg)

        self.assertEqual(len(outs), 4)
        self.assertEqual(tuple(outs[0].shape), (4, 1, 30, 30))
        self.assertEqual(tuple(outs[1].shape), (4, 1, 30, 30))
        self.assertEqual(tuple(outs[2].shape), (4, 1, 14, 14))
        self.assertEqual(tuple(outs[3].shape), (4, 1, 6, 6))


if __name__ == '__main__':
    unittest.main()
