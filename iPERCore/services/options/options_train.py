import os
from .options_base import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self._parser.add_argument("--dataset_mode", type=str, default="iPER", help="chooses dataset to be used")
        self._parser.add_argument("--train_ids_file", type=str, default="train.txt", help="file containing train ids")
        self._parser.add_argument("--test_ids_file", type=str, default="val.txt", help="file containing test ids")

        # use place dataset if need
        self._parser.add_argument("--place_dir", type=str, default="/p300/places365_standard", help="place folder")

        # use iPER dataset
        self._parser.add_argument("--iPER_dir", type=str, default="", help="iPER dataset folder")

        # use deep fashion dataset if need
        self._parser.add_argument("--fashion_dir", type=str, default="", help="place folder")

        # use deep fashion dataset if need
        self._parser.add_argument("--motion_synthetic_dir", type=str, default="", help="motion synthetic folder")

        self.is_train = True

    def parse(self):
        cfg = super().parse()
        checkpoints_dir = cfg.meta_data.checkpoints_dir
        cfg = self.set_and_check_load_epoch(cfg, checkpoints_dir)

        return cfg

    def set_and_check_load_epoch(self, cfg, checkpoints_dir):
        if os.path.exists(checkpoints_dir):
            if cfg.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(checkpoints_dir):
                    if file.startswith("net_epoch_"):
                        epoch_name = file.split("_")[2]
                        if epoch_name.isdigit():
                            load_epoch = max(load_epoch, int(epoch_name))
                cfg.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(checkpoints_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split("_")[2]) == cfg.load_epoch
                        if found: break
                assert found, f"Model for epoch {cfg.load_epoch} not found"
        else:
            assert cfg.load_epoch < 1, f"Model for epoch {cfg.load_epoch} not found"
            cfg.load_epoch = 0

        return cfg
