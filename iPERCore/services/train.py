# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import time
from collections import OrderedDict

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


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Train(object):
    def __init__(self, args):
        self._setup(args)
        self._train()

    def _setup(self, args):
        if args.use_cudnn:
            # cudnn related setting
            cudnn.benchmark = True
            cudnn.deterministic = False
            cudnn.enabled = True

        gpus = args.gpu_ids
        distributed = len(gpus) > 1
        device = torch.device("cuda:{}".format(args.local_rank))

        if distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )

        torch.autograd.set_detect_anomaly(True)

        # prepare data
        train_dataset = DatasetFactory.get_by_name(args.dataset_mode, args, is_for_train=True)

        if distributed:
            train_sampler = DistributedSampler(train_dataset)
            shuffle = False
        else:
            train_sampler = None
            shuffle = not args.serial_batches

        # TODO: worker_init_fn
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler)

        test_dataset = DatasetFactory.get_by_name(args.dataset_mode, args, is_for_train=False)

        # TODO: worker_init_fn
        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True)

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

    def _train(self):
        self._total_steps = self._opt.load_epoch * self._train_size
        # self._iters_per_epoch = self._train_size // (self._opt.batch_size * self._num_gpus)
        self._iters_per_epoch = len(self._trainloader)
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()

        for i_epoch in range(self._opt.load_epoch + 1,
                             self._opt.Train.niters_or_epochs_no_decay + self._opt.Train.niters_or_epochs_decay + 1):
            epoch_start_time = time.time()

            # train epoch
            self._train_epoch(i_epoch)

            if self._check_is_major_rank():
                # save model
                print("saving the model at the end of epoch %d, iters %d" % (i_epoch, self._total_steps))
                self._model.save(i_epoch)

                # print epoch info
                time_epoch = time.time() - epoch_start_time
                print("End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)" %
                      (i_epoch, self._opt.Train.niters_or_epochs_no_decay + self._opt.Train.niters_or_epochs_decay, 
                       time_epoch, time_epoch / 60, time_epoch / 3600))

            # update learning rate
            if self._check_need_update_lr(i_epoch):
                self._model.update_learning_rate()

    def _check_is_major_rank(self):
        return self._opt.local_rank == 0

    def _check_need_update_lr(self, i_epoch):
        return i_epoch >= self._opt.Train.niters_or_epochs_no_decay and \
            i_epoch != self._opt.Train.niters_or_epochs_no_decay + self._opt.Train.niters_or_epochs_decay

    def _train_epoch(self, i_epoch):
        is_major_rank = self._check_is_major_rank()
        epoch_iter = 0
        self._model.set_train()
        for i_train_batch, train_batch in enumerate(self._trainloader):
            iter_start_time = time.time()

            # display flags
            do_visuals = self._last_display_time is None or iter_start_time - self._last_display_time > self._opt.Train.display_freq_s
            do_print_terminal = iter_start_time - self._last_print_time > self._opt.Train.print_freq_s or do_visuals
            do_save = (self._last_save_latest_time is None) or \
                      (iter_start_time - self._last_save_latest_time > self._opt.Train.save_latest_freq_s)

            # train model
            self._model.set_input(train_batch, self._device)
            trainable = (i_train_batch+1) % self._opt.Train.train_G_every_n_iterations == 0
            self._model.optimize_parameters(keep_data_for_visuals=do_visuals and is_major_rank, trainable=trainable)

            # update epoch info
            self._total_steps += self._opt.batch_size
            epoch_iter += self._opt.batch_size

            if is_major_rank:
                iter_end_time = time.time()

                # display terminal
                if do_print_terminal:
                    self._display_terminal(iter_start_time, i_epoch, i_train_batch, do_visuals)
                    self._last_print_time = iter_end_time

                # display visualizer
                if do_visuals:
                    print("visualizing on Tensorboard.")
                    self._display_visualizer_train(self._total_steps)
                    self._display_visualizer_val(i_epoch, self._total_steps)
                    self._last_display_time = iter_end_time

                if do_save:
                    # save model
                    print("saving the model at the end of epoch %d, iters %d" % (i_epoch, self._total_steps))
                    self._model.save(i_epoch)
                    self._last_save_latest_time = iter_end_time

    def _display_terminal(self, iter_start_time, i_epoch, i_train_batch, visuals_flag):
        errors = self._model.get_current_errors()
        t = (time.time() - iter_start_time) / self._opt.batch_size
        self._tb_visualizer.print_current_train_errors(i_epoch, i_train_batch, self._iters_per_epoch, errors, t, visuals_flag)

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
