# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

"""

This script is the wrapper to the actual training script `iPERCore/services/train.py`.

usage: iPERCore/services/train.py [-h] [--cfg_path CFG_PATH] [--verbose]
                [--num_source NUM_SOURCE] [--image_size IMAGE_SIZE]
                [--batch_size BATCH_SIZE] [--time_step TIME_STEP]
                [--intervals INTERVALS] [--load_epoch LOAD_EPOCH]
                [--bg_ks BG_KS] [--ft_ks FT_KS] [--only_vis] [--temporal]
                [--use_inpaintor] [--gpu_ids GPU_IDS] [--use_cudnn]
                [--output_dir OUTPUT_DIR] [--model_id MODEL_ID]
                [--dataset_mode DATASET_MODE]
                [--dataset_dirs [DATASET_DIRS [DATASET_DIRS ...]]]
                [--background_dir BACKGROUND_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --cfg_path CFG_PATH   the configuration path. (default:
                        ./assets/configs/deploy.toml)
  --verbose             print the options or not. (default: False)
  --num_source NUM_SOURCE
                        number of source (default: 2)
  --image_size IMAGE_SIZE
                        input image size (default: 512)
  --batch_size BATCH_SIZE
                        input batch size (default: 1)
  --time_step TIME_STEP
                        time step size (default: 1)
  --intervals INTERVALS
                        the interval between frames. (default: 1)
  --load_epoch LOAD_EPOCH
                        which epoch to load? set to -1 to use latest cached
                        model (default: -1)
  --bg_ks BG_KS         dilate kernel size of background mask. (default: 11)
  --ft_ks FT_KS         dilate kernel size of front mask. (default: 1)
  --only_vis            only visible or not (default: False)
  --temporal            use temporal warping or not (default: False)
  --use_inpaintor       if there is no background, use additional background
                        inpaintor network, such as deepfillv2 to get the
                        background image. (default: False)
  --gpu_ids GPU_IDS     gpu ids: e.g. 0 0,1,2, 0,2. (default: 0)
  --use_cudnn           whether to use cudnn or not, if true, do not use.
                        (default: False)
  --output_dir OUTPUT_DIR
                        the data directory, it contains --data_dir/primitives,
                        this directory to save the processed and synthesis,
                        --data_dir/models, this directory to save the models
                        and summaries. (default: ./results)
  --model_id MODEL_ID   name of the checkpoints directory. The model will be
                        saved in output_dir/models/model_id. (default:
                        default)
  --dataset_mode DATASET_MODE
                        chooses dataset to be used (default: ProcessedVideo)
  --dataset_dirs [DATASET_DIRS [DATASET_DIRS ...]]
                        the directory of all processed datasets. (default:
                        ['/p300/tpami/datasets/fashionvideo',
                        '/p300/tpami/datasets/iPER',
                        '/p300/tpami/datasets/motionSynthetic'])
  --background_dir BACKGROUND_DIR
                        the directory of background inpainting dataset, e.g
                        Place2. (default: /p300/places365_standard)

"""

import subprocess
import os
import sys


##################################### args for training #######################################
gpu_ids = "0,1,2,3"             # distributed multi-GPUs
# gpu_ids = "0"                 # single GPU
distributed_port = "10086"
cfg_path = "./assets/configs/trainers/train_aug_bg.toml"
output_dir = "/p300/tpami/checkpoints"

# Use iPER + MotionSynthetic + FashionVideo + Place2 datasets
model_id = "iPER+MS+FashionVideo+Place2"
dataset_dirs = ["/p300/tpami/datasets/fashionvideo",
                "/p300/tpami/datasets/iPER",
                "/p300/tpami/datasets/motionSynthetic"]       ## replace this with your path
background_dir = "/p300/tpami/places"                         ## replace this with your path
dataset_mode = "ProcessedVideo+Place2"

## Use iPER + Place2 datasets
# model_id = "iPER+Place2"
# dataset_dirs = ["/p300/tpami/datasets/iPER"]                  ## replace this with your path
# dataset_mode = "ProcessedVideo+Place2"                        ## replace this with your path

## Use iPER + Place2 datasets
# model_id = "iPER+Place2"
# dataset_dirs = ["/p300/tpami/datasets/iPER"]                  ## replace this with your path
# dataset_mode = "ProcessedVideo+Place2"                        ## replace this with your path

## Use MotionSynthetic + Place2 datasets
# model_id = "MS+Place2"
# dataset_dirs = ["/p300/tpami/datasets/motionSynthetic"]       ## replace this with your path
# background_dir = "/p300/tpami/places"                         ## replace this with your path
# dataset_mode = "ProcessedVideo+Place2"

## Use iPER dataset
# model_id = "iPER"
# dataset_dirs = ["/p300/tpami/datasets/iPER",                  ## replace this with your path
#                 "/p300/tpami/datasets/motionSynthetic"]       ## replace this with your path
# dataset_mode = "ProcessedVideo"

image_size = 512
num_source = 4
time_step = 2
batch_size = 1
################################################################################################


################################### wrapper to call iPERCore/services/train.py #################
nproc_per_node = len(gpu_ids.split(","))
num_cores = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores // nproc_per_node)


cmd = [
    sys.executable, "-m", "torch.distributed.launch", "--master_port", distributed_port,
    "--nproc_per_node", str(nproc_per_node), "-m", "iPERCore.services.train",
    "--gpu_ids",        gpu_ids,
    "--cfg_path",       cfg_path,
    "--model_id",       model_id,
    "--output_dir",     output_dir,
    "--model_id",       model_id,
    "--dataset_mode",   dataset_mode,
    "--dataset_dirs",   " ".join(dataset_dirs),
    "--background_dir", background_dir,
    "--image_size",     str(image_size),
    "--num_source",     str(num_source),
    "--time_step",      str(time_step),
    "--batch_size",     str(batch_size),
    "--use_cudnn"
]

subprocess.call(" ".join(cmd), shell=True)




