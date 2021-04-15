# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
from .options_base import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self._parser.add_argument("--dataset_mode", type=str, default="ProcessedVideo",
                                  choices=["ProcessedVideo", "ProcessedVideo+Place2"],
                                  help="chooses dataset to be used.")

        self._parser.add_argument("--dataset_dirs", type=str, nargs="*",
                                  default=["/p300/tpami/datasets/fashionvideo",
                                           "/p300/tpami/datasets/iPER",
                                           "/p300/tpami/datasets/motionSynthetic"],
                                  help="the directory of all processed datasets.")

        # use place dataset if need
        self._parser.add_argument("--background_dir", type=str, default="/p300/places365_standard",
                                  help="the directory of background inpainting dataset, e.g Place2.")

        self.is_train = True

    def parse(self):
        cfg = super().parse()
        checkpoints_dir = cfg.meta_data.checkpoints_dir
        cfg = self.set_and_check_load_iter(cfg, checkpoints_dir)

        return cfg

    def set_and_check_load_iter(self, cfg, checkpoints_dir):
        if os.path.exists(checkpoints_dir):
            if cfg.load_iter == -1:
                load_iter = 0
                for file in os.listdir(checkpoints_dir):
                    if file.startswith("net_iter_"):
                        epoch_name = file.split("_")[2]
                        if epoch_name.isdigit():
                            load_iter = max(load_iter, int(epoch_name))
                cfg.load_iter = load_iter
            else:
                found = False
                for file in os.listdir(checkpoints_dir):
                    if file.startswith("net_iter_"):
                        found = int(file.split("_")[2]) == cfg.load_iter
                        if found: break
                assert found, f"Model for epoch {cfg.load_iter} not found"
        else:
            assert cfg.load_iter < 1, f"Model for epoch {cfg.load_iter} not found"
            cfg.load_iter = 0

        return cfg
