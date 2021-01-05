# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import unittest
import torch
import toml
from easydict import EasyDict

from iPERCore.models.networks import generators


class TestGenerators(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.attlwb_cfg_str = """
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
                norm_type = "instance"            # use instance normalization
                ndf = 64            # number of the base filters for Discriminator.
                n_down = 6          # number of conditions, RGB (3) + UV_Seg (3) = 6

        """

    def test_01_AttentionLWBGenerator(self):

        cfg_str = self.attlwb_cfg_str
        cfg = EasyDict(toml.loads(cfg_str))

        gen_cfg = cfg["Generator"]

        generator = generators.AttentionLWBGenerator(gen_cfg, temporal=False)

        bg_inputs = torch.rand(4, 5, 4, 512, 512)
        src_inputs = torch.rand(4, 5, 6, 512, 512)
        tsf_inputs = torch.rand(4, 2, 6, 512, 512)
        Tst = torch.rand(4, 2, 5, 512, 512, 2)
        Ttt = torch.rand(4, 1, 512, 512, 2)

        bg_img, src_img, src_mask, tsf_img, tsf_mask = generator(
            bg_inputs, src_inputs, tsf_inputs, Tst, Ttt,
            only_tsf=False
        )

        print(bg_img.shape, src_img.shape, src_mask.shape, tsf_img.shape, tsf_mask.shape)

        self.assertEqual(tuple(bg_img.shape), (4, 5, 3, 512, 512))
        self.assertEqual(tuple(src_img.shape), (4, 5, 3, 512, 512))
        self.assertEqual(tuple(src_mask.shape), (4, 5, 1, 512, 512))
        self.assertEqual(tuple(tsf_img.shape), (4, 2, 3, 512, 512))
        self.assertEqual(tuple(tsf_mask.shape), (4, 2, 1, 512, 512))

    def test_02_AttentionLWBFrontGenerator(self):

        cfg_str = self.attlwb_cfg_str
        cfg = EasyDict(toml.loads(cfg_str))

        gen_cfg = cfg["Generator"]

        generator = generators.AttentionLWBFrontGenerator(gen_cfg, temporal=False)

        src_inputs = torch.rand(4, 5, 6, 512, 512)
        tsf_inputs = torch.rand(4, 2, 6, 512, 512)
        Tst = torch.rand(4, 2, 5, 512, 512, 2)
        Ttt = torch.rand(4, 1, 512, 512, 2)

        src_img, src_mask, tsf_img, tsf_mask = generator(
            src_inputs, tsf_inputs, Tst, Ttt,
            only_tsf=False
        )

        print(src_img.shape, src_mask.shape, tsf_img.shape, tsf_mask.shape)

        self.assertEqual(tuple(src_img.shape), (4, 5, 3, 512, 512))
        self.assertEqual(tuple(src_mask.shape), (4, 5, 1, 512, 512))
        self.assertEqual(tuple(tsf_img.shape), (4, 2, 3, 512, 512))
        self.assertEqual(tuple(tsf_mask.shape), (4, 2, 1, 512, 512))


if __name__ == '__main__':
    unittest.main()
