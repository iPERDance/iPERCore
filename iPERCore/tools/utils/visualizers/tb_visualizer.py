# https://github.com/albertpumarola/GANimation/blob/master/utils/tb_visualizer.py

import os
import time
from tensorboardX import SummaryWriter

from iPERCore.tools.utils.filesio.cv_utils import save_cv2_img


class TBVisualizer(object):
    def __init__(self, opt):
        self._opt = opt
        self._save_path = opt.meta_data.checkpoints_dir

        self._log_path = os.path.join(self._save_path, "loss_log2.txt")
        self._tb_path = os.path.join(self._save_path, "summary.json")
        self._writer = SummaryWriter(self._save_path)

        with open(self._log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write("================ Training Loss (%s) ================\n" % now)

    def __del__(self):
        self._writer.close()

    @property
    def log_path(self):
        return self._log_path

    @property
    def tb_path(self):
        return self._tb_path

    @property
    def writer(self):
        return self._writer

    def display_current_results(self, visuals, it, is_train, save_visuals=False):
        for label, image_numpy in visuals.items():
            # ipdb.set_trace()
            sum_name = "{}/{}".format("Train" if is_train else "Test", label)
            self._writer.add_image(sum_name, image_numpy, it)

            if save_visuals:
                save_cv2_img(image_numpy,
                             os.path.join(self._opt.checkpoints_dir, self._opt.name,
                                          "event_imgs", sum_name, "%08d.png" % it))

        self._writer.export_scalars_to_json(self._tb_path)

    def plot_scalars(self, scalars, it, is_train):
        for label, scalar in scalars.items():
            sum_name = "{}/{}".format("Train" if is_train else "Test", label)
            self._writer.add_scalar(sum_name, scalar, it)

    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, t, visuals_were_stored):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        visuals_info = "v" if visuals_were_stored else ""
        message = "%s (T%s, epoch: %d, it: %d/%d, t/smpl: %.3fs)\n" % (
        log_time, visuals_info, epoch, i, iters_per_epoch, t)
        for k, v in errors.items():
            msg = "\t%s:%.3f\n" % (k, v)
            message += msg
        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write("%s\n" % message)

    def print_current_validate_errors(self, epoch, errors, t):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = "%s (V, epoch: %d, time_to_val: %ds)\n" % (log_time, epoch, t)
        for k, v in errors.items():
            message += "\t%s:%.3f\n" % (k, v)

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write("%s\n" % message)

    def save_images(self, visuals):
        for label, image_numpy in visuals.items():
            image_name = "%s.png" % label
            save_path = os.path.join(self._save_path, "samples", image_name)
            save_cv2_img(image_numpy, save_path)
