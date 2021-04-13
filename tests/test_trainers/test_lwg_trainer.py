# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import unittest
import torch
from easydict import EasyDict

from iPERCore.services.options.options_setup import setup
from iPERCore.tools.trainers.lwg_trainer import LWGTrainer


class TrainerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg_path = "./assets/configs/deploy.toml"

    def test_01_lwg_trainer(self):
        opt = EasyDict()

        opt.cfg_path = self.cfg_path
        opt.gpu_ids = "2"
        opt.model_id = "debug"
        opt.output_dir = "../tests/data"
        opt.verbose = True
        opt.temporal = False
        opt.load_iter = -1

        cfg = setup(opt)

        device = torch.device("cuda:{}".format(cfg.local_rank))

        lwg_trainer = LWGTrainer(cfg, device)
        lwg_trainer.gpu_wrapper()


if __name__ == '__main__':
    unittest.main()
