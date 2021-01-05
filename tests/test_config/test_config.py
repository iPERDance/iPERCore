# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import unittest
import os.path as osp
import toml
from easydict import EasyDict
import pprint


data_dir = osp.join(osp.dirname(osp.dirname(__file__)), "data")
config_dir = osp.join(data_dir, "configs")
config_path = osp.join(config_dir, "default.toml")


class TestConfig(unittest.TestCase):

    def test_01_load(self):

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = toml.load(f, _dict=EasyDict)

        pprint.pprint(cfg)


if __name__ == '__main__':
    unittest.main()
