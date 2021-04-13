# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import argparse

from .options_setup import setup


class BaseOptions(object):
    def __init__(self):
        self._parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self._initialized = False
        self.is_train = False

    def initialize(self):
        # configs
        self._parser.add_argument("--cfg_path", type=str, default="./assets/configs/deploy.toml",
                                  help="the configuration path.")
        self._parser.add_argument("--verbose", action="store_true", help="print the options or not.")

        self._parser.add_argument("--num_source", type=int, default=2, help="number of source")
        self._parser.add_argument("--image_size", type=int, default=512, help="input image size")
        self._parser.add_argument("--batch_size", type=int, default=1, help="input batch size")
        self._parser.add_argument("--time_step", type=int, default=1, help="time step size")
        self._parser.add_argument("--intervals", type=int, default=1, help="the interval between frames.")
        self._parser.add_argument("--load_iter", type=int, default=-1,
                                  help="which epoch to load? set to -1 to use latest cached model")

        self._parser.add_argument("--bg_ks", default=11, type=int, help="dilate kernel size of background mask.")
        self._parser.add_argument("--ft_ks", default=1, type=int, help="dilate kernel size of front mask.")

        self._parser.add_argument("--only_vis", action="store_true",
                                  default=False, help="only visible or not")
        self._parser.add_argument("--temporal", action="store_true",
                                  default=False, help="use temporal warping or not")
        self._parser.add_argument("--use_inpaintor", action="store_true",
                                  default=False, help="if there is no background, use additional background inpaintor "
                                                      "network, such as deepfillv2 to get the background image.")

        # gpu settings
        self._parser.add_argument("--gpu_ids", type=str, default="0",
                                  help="gpu ids: e.g. 0  0,1,2, 0,2.")
        self._parser.add_argument("--local_rank", type=int, default=0,
                                  help="the local rank for distributed training.")
        self._parser.add_argument("--use_cudnn", action="store_true",
                                  help="whether to use cudnn or not, if true, do not use.")

        # meta-data settings
        self._parser.add_argument("--output_dir", type=str, default="./results",
                                  help="the data directory, it contains \n"
                                       "--data_dir/primitives, this directory to save the processed and synthesis,\n"
                                       "--data_dir/models, this directory to save the models and summaries.")
        self._parser.add_argument("--model_id", type=str, default="default",
                                  help="name of the checkpoints directory. "
                                       "The model will be saved in output_dir/models/model_id.")

        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()

        # opt = self._parser.parse_args()
        opt, extra_args = self._parser.parse_known_args()
        opt.is_train = self.is_train

        cfg = setup(opt, extra_args)

        return cfg
