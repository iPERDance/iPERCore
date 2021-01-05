# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import unittest
import torch

from iPERCore.tools.human_pose3d_estimators.spin import SPINRunner


class TestSPINRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.cfg_or_path = "./assets/configs/pose3d/spin.toml"
        cls.device = torch.device("cuda:0")
        cls.spin = SPINRunner(cfg_or_path=cls.cfg_or_path, device=cls.device)

    def test_01(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
