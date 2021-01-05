# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import unittest
import os.path as osp

from iPERCore.tools.human_mattors import build_mattor


data_dir = osp.join(osp.dirname(osp.dirname(__file__)), "data")
config_dir = osp.join(data_dir, "configs")
config_path = osp.join(config_dir, "default.toml")


class TestPointRenderParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.default_cfg = None

        cls.inpaintor = None

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
