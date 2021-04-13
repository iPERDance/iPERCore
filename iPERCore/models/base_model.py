# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import torch
from collections import OrderedDict


class ModelsFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, *args, **kwargs):
        model = None

        if model_name == "imitator":
            from .imitator import Imitator
            model = Imitator(*args, **kwargs)

        elif model_name == "swapper":
            from .imitator import Swapper
            model = Swapper(*args, **kwargs)

        elif model_name == "viewer":
            from .imitator import Viewer
            model = Viewer(*args, **kwargs)

        else:
            raise ValueError(f"Model {model_name} not recognized.")

        print(f"Model {model.name} was created")
        return model


class BaseModel(object):
    def __init__(self, opt):
        self._name = "BaseModel"

        self._opt = opt
        self._save_dir = opt.meta_data.checkpoints_dir

    @property
    def name(self):
        return self._name

    def load_network(self, network, network_label, epoch_label, need_module=False):
        load_filename = "net_iter_%s_id_%s.pth" % (epoch_label, network_label)
        load_path = os.path.join(self._save_dir, load_filename)

        self.load_params(network, load_path, need_module)

    def load_params(self, network, load_path, need_module=False):
        assert os.path.exists(
            load_path), "Weights file not found. Have you trained a model!? We are not providing one %s" % load_path

        def load(model, orig_state_dict):
            state_dict = OrderedDict()
            for k, v in orig_state_dict.items():
                # remove "module"
                name = k[7:] if "module" in k else k
                state_dict[name] = v

            # load params
            # model.load_state_dict(state_dict)
            model.load_state_dict(state_dict, strict=False)

        save_data = torch.load(load_path, map_location="cpu")
        if need_module:
            # network.load_state_dict(save_data)
            network.load_state_dict(save_data, strict=False)
        else:
            load(network, save_data)

        print("Loading net: %s" % load_path)


class BaseRunnerModel(BaseModel):

    def __init__(self, opt):
        super(BaseRunnerModel, self).__init__(opt)

        self._name = "BaseRunnerModel"

    def source_setup(self, *args, **kwargs):
        raise NotImplementedError

    def swap_params(self, *args, **kwargs):
        raise NotImplementedError

    def make_inputs_for_tsf(self, *args, **kwargs):
        raise NotImplementedError

    def post_update(self, *args, **kwargs):
        raise NotImplementedError

