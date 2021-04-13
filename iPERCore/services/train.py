# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import time
from collections import OrderedDict
import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.distributed
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler

from iPERCore.tools.utils.visualizers.tb_visualizer import TBVisualizer
from iPERCore.data.dataset import DatasetFactory
from iPERCore.tools.trainers import create_trainer


def set_cudnn():
    # cudnn related setting
    cudnn.benchmark = True
    cudnn.deterministic = False
    # cudnn.deterministic = True
    cudnn.enabled = True


def worker_init_fn(worker_id):
    worker_seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(worker_seed)


class Train(object):
    def __init__(self, args):
        self._setup(args)
        self._train()

    def _setup(self, args):

        if args.use_cudnn:
            set_cudnn()

        gpus = args.gpu_ids
        distributed = len(gpus) > 1
        device = torch.device("cuda:{}".format(args.local_rank))

        if distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )

        # torch.autograd.set_detect_anomaly(True)

        # prepare data
        train_dataset = DatasetFactory.get_by_name(args.dataset_mode, args, is_for_train=True)

        if distributed:
            train_sampler = DistributedSampler(train_dataset)
            shuffle = False
        else:
            train_sampler = None
            shuffle = not args.serial_batches

        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler,
            worker_init_fn=worker_init_fn
        )

        test_dataset = DatasetFactory.get_by_name(args.dataset_mode, args, is_for_train=False)
        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        # build model trainer
        model_trainer = create_trainer(args.train_name, args, device)

        if distributed:
            def _transform_(m):
                return nn.parallel.DistributedDataParallel(
                    m, device_ids=[args.local_rank], output_device=args.local_rank,
                    find_unused_parameters=True
                )
            model_trainer.multi_gpu_wrapper(_transform_)
        else:
            model_trainer.gpu_wrapper()

        self._opt = args
        self._num_gpus = len(gpus)
        self._num_per_iter = self._num_gpus * args.batch_size
        self._device = device
        self._model = model_trainer
        self._trainloader = trainloader
        self._testloader = testloader
        self._train_size = len(train_dataset)
        self._test_size = len(test_dataset)
        self._tb_visualizer = TBVisualizer(args)

        if args.local_rank == 0:
            print("#train video clips = %d" % self._train_size)
            print("#test video clips = %d" % self._test_size)
            pprint.pprint(args)

    def check_do_visuals(self, iter_start_time):
        do_visuals = (self._last_display_time is None
                      or iter_start_time - self._last_display_time > self._opt.Train.display_freq_s) and (
                         self._tb_visualizer is not None)
        return do_visuals

    def check_print_terminal(self, iter_start_time, do_visuals):
        do_print_terminal = (iter_start_time - self._last_print_time > self._opt.Train.print_freq_s or do_visuals) \
                            and (self._tb_visualizer is not None)
        return do_print_terminal

    def check_save_model(self, iter_start_time):
        do_save = (self._last_save_latest_time is None
                   or iter_start_time - self._last_save_latest_time > self._opt.Train.save_latest_freq_s) and (
                      self._tb_visualizer is not None)
        return do_save

    def check_is_local(self):
        # check is local rank
        is_local_rank = self._opt.local_rank == 0
        return is_local_rank

    def _train(self):
        # self._iters_per_epoch = self._train_size // (self._opt.batch_size * self._num_gpus)
        self._iters_per_epoch = len(self._trainloader)
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()

        total_steps = self._opt.load_iter
        self._total_iters = self._opt.Train.niters_or_epochs_no_decay + self._opt.Train.niters_or_epochs_decay

        i_epoch = 0
        while total_steps < self._total_iters:
            i_epoch += 1
            for i_train_batch, train_batch in enumerate(self._trainloader):
                iter_start_time = time.time()

                # check is local rank
                is_local_rank = self.check_is_local()

                # display flags
                do_visuals = self.check_do_visuals(iter_start_time)
                do_print_terminal = self.check_print_terminal(iter_start_time, do_visuals)
                do_save = self.check_save_model(iter_start_time)

                # train model
                self._model.set_input(train_batch, self._device)
                trainable = (i_train_batch + 1) % self._opt.Train.train_G_every_n_iterations == 0
                self._model.optimize_parameters(trainable=trainable, keep_data_for_visuals=do_visuals)

                # update epoch info
                total_steps += self._num_per_iter

                # display terminal
                if is_local_rank and do_print_terminal:
                    self._display_terminal(iter_start_time, i_epoch, total_steps, do_visuals)
                    self._last_print_time = iter_start_time

                # display visualizer
                if is_local_rank and do_visuals:
                    self._display_visualizer_train(total_steps)
                    self._display_visualizer_val(i_epoch, total_steps)
                    self._last_display_time = iter_start_time

                # save checkpoints
                if is_local_rank and do_save:
                    print(f"saving the model at the end of epoch %d, iters %d" % (i_epoch, total_steps))
                    self._model.save(total_steps)
                    self._last_save_latest_time = iter_start_time

                if total_steps >= self._total_iters:
                    break

        if self.check_is_local():
            print(f"saving the model at the end of epoch %d, iters %d" % (i_epoch, self._total_iters))
            self._model.save(self._total_iters)

    def _check_is_major_rank(self):
        return self._opt.local_rank == 0

    def _check_need_update_lr(self, i_epoch):
        return i_epoch >= self._opt.Train.niters_or_epochs_no_decay and \
               i_epoch != self._opt.Train.niters_or_epochs_no_decay + self._opt.Train.niters_or_epochs_decay

    def _display_terminal(self, iter_start_time, i_epoch, step, visuals_flag):
        errors = self._model.get_current_errors()
        t = (time.time() - iter_start_time) / self._opt.batch_size
        self._tb_visualizer.print_current_train_errors(i_epoch, step, self._total_iters, errors, t, visuals_flag)

    def _display_visualizer_train(self, total_steps):
        self._tb_visualizer.display_current_results(self._model.get_current_visuals(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_errors(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_scalars(), total_steps, is_train=True)

    def _display_visualizer_val(self, i_epoch, total_steps):
        val_start_time = time.time()

        # set model to eval
        self._model.set_eval()

        # evaluate self._opt.num_iters_validate epochs
        val_errors = OrderedDict()
        for i_val_batch, val_batch in enumerate(self._testloader):
            if i_val_batch == self._opt.Train.num_iters_validate:
                break

            # evaluate model
            self._model.set_input(val_batch, self._device)
            self._model.forward(keep_data_for_visuals=(i_val_batch == 0))
            errors = self._model.get_current_errors()

            # store current batch errors
            for k, v in errors.items():
                if k in val_errors:
                    val_errors[k] += v
                else:
                    val_errors[k] = v

        # normalize errors
        for k in val_errors:
            val_errors[k] /= self._opt.Train.num_iters_validate

        # visualize
        t = (time.time() - val_start_time)
        self._tb_visualizer.print_current_validate_errors(i_epoch, val_errors, t)
        self._tb_visualizer.plot_scalars(val_errors, total_steps, is_train=False)
        self._tb_visualizer.display_current_results(self._model.get_current_visuals(), total_steps, is_train=False)

        # set model back to train
        self._model.set_train()


if __name__ == "__main__":
    from iPERCore.services.options.options_train import TrainOptions

    cfg = TrainOptions().parse()
    Train(cfg)
