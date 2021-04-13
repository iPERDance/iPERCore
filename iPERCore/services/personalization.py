# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import time
import numpy as np
import torch
import torch.distributed
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from multiprocessing import Process

from iPERCore.data.personalized_dataset import PersonalizedDataset
from iPERCore.tools.utils.visualizers.tb_visualizer import TBVisualizer
from iPERCore.tools.trainers import create_trainer


__all__ = ["PersonalizerProcess", "personalize"]


def set_cudnn():
    # cudnn related setting
    cudnn.benchmark = True
    # cudnn.deterministic = False
    cudnn.deterministic = True
    cudnn.enabled = True


def worker_init_fn(worker_id):
    worker_seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(worker_seed)


class PersonalizerProcess(Process):
    def __init__(self, opt):
        """

        Args:
            opt:
        """

        # 1. setup the seed of numpy
        np.random.seed(2020)

        # 2. set gpu devices
        gpus = opt.gpu_ids
        device = torch.device("cuda:{}".format(opt.local_rank))

        # 3. prepare dataset and dataloader
        train_dataset = PersonalizedDataset(opt, opt.meta_data.meta_src)

        trainloader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )

        self._opt = opt
        self._num_gpus = len(gpus)
        self._device = device
        self._model = None   # this will be initialized in self.run()
        self._num_videos = train_dataset.num_videos
        self._train_size = len(train_dataset)
        self._trainloader = trainloader
        self._iters_per_epoch = len(self._trainloader)
        self._last_print_time = None
        self._last_display_time = None

        if opt.tb_visual:
            self._tb_visualizer = TBVisualizer(opt)
        else:
            self._tb_visualizer = None

        print(f"#train video clips = {train_dataset.size()}")

        super().__init__(name=f"Personalizer_{opt.gpu_ids}")

    def check_do_visuals(self, iter_start_time):
        do_visuals = (self._last_display_time is None
                      or iter_start_time - self._last_display_time > self._opt.Train.display_freq_s) and (
                         self._tb_visualizer is not None)
        return do_visuals

    def check_print_terminal(self, iter_start_time, do_visuals):
        do_print_terminal = (iter_start_time - self._last_print_time > self._opt.Train.print_freq_s or do_visuals) \
                            and (self._tb_visualizer is not None)
        return do_print_terminal

    def run(self) -> None:
        self._last_display_time = None
        self._last_print_time = time.time()

        i_epoch = 0
        total_steps = 0
        total_iters = (self._opt.Train.niters_or_epochs_no_decay +
                       self._opt.Train.niters_or_epochs_decay) * self._num_videos

        progressbar = tqdm(total=total_iters)

        # use cudnn or not
        if self._opt.use_cudnn:
            set_cudnn()

        # build model trainer
        model_trainer = create_trainer(self._opt.train_name, self._opt, self._device)
        self._model = model_trainer.gpu_wrapper()
        self._model.set_train()

        while total_steps < total_iters:
            epoch_iter = 0
            i_epoch += 1
            for i_train_batch, train_batch in enumerate(self._trainloader):
                iter_start_time = time.time()

                # display flags
                do_visuals = self.check_do_visuals(iter_start_time)
                do_print_terminal = self.check_print_terminal(iter_start_time, do_visuals)

                # train model
                self._model.set_input(train_batch, self._device)
                trainable = (i_train_batch + 1) % self._opt.Train.train_G_every_n_iterations == 0
                self._model.optimize_parameters(trainable=trainable, keep_data_for_visuals=do_visuals)

                # update epoch info
                progressbar.update()
                total_steps += 1
                epoch_iter += 1

                # display terminal
                if do_print_terminal:
                    self._display_terminal(iter_start_time, i_epoch, i_train_batch, do_visuals)
                    self._last_print_time = iter_start_time

                # display visualizer
                if do_visuals:
                    self._display_visualizer_train(total_steps)
                    self._last_display_time = iter_start_time

                if total_steps >= total_iters:
                    break

        progressbar.close()

        torch.save(self._model.G.state_dict(), self._opt.meta_data.personalized_ckpt_path)
        print(f"saving the personalized model in {self._opt.meta_data.personalized_ckpt_path}")

    def _display_terminal(self, iter_start_time, i_epoch, i_train_batch, visuals_flag):
        errors = self._model.get_current_errors()
        t = (time.time() - iter_start_time) / self._opt.batch_size
        self._tb_visualizer.print_current_train_errors(i_epoch, i_train_batch,
                                                       self._iters_per_epoch, errors, t, visuals_flag)

    def _display_visualizer_train(self, total_steps):
        self._tb_visualizer.display_current_results(self._model.get_current_visuals(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_errors(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_scalars(), total_steps, is_train=True)


def personalize(opt):
    """

    Args:
        opt:

    Returns:

    """

    print("Step 2: running personalization on")

    personalized_ckpt_path = opt.meta_data.personalized_ckpt_path

    if not os.path.exists(personalized_ckpt_path):
        personalizer = PersonalizerProcess(opt)
        personalizer.start()
        personalizer.join()

    print(f"Step 2: personalization done, saved in {personalized_ckpt_path}...")


if __name__ == "__main__":
    from iPERCore.services.options.options_inference import InferenceOptions
    OPT = InferenceOptions().parse()

    personalize(OPT)

