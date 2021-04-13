# Training Details

## Prepare all datasets

### 1. Prepare iPER dataset

* `--output_dir`: the $iPER_root_dir;
* `--gpu_ids`: the gpu ids. The preprocessor needs at least `3.2G` GPU memory.
If we have a RTX 2080Ti with `11G`, then each GPU could run 3 preprocessors. For example,
  we have two RTX 2080Ti(s), and we can set this as `--gpu_ids 0,0,0,1,1,1` with 6 parallel preprocessors.
  
```Bash
python scripts/train/prepare_iPER_dataset.py \
   --output_dir /p300/tpami/datasets_reproduce/iPER \
   --gpu_ids 0,0,0,1,1,1,2,2,2,3,3,3,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9
```

**If it encounters with some errors during the downloading stage, manually downloading all stuffs and 
referring to [manually_download_datasets.md](manually_download_datasets.md) for more details.**

### 3.Prepare MotionSynthetic dataset
* `--output_dir`: the $motionSynthetic_root_dir;
* `--gpu_ids`: see above.

```Bash
python scripts/train/prepare_motionSynthetic_dataset.py \
   --output_dir /p300/tpami/datasets_reproduce/motionSynthetic \
   --gpu_ids 0,0,0,0,1,1,1,1
```

**If it encounters with some errors during the downloading stage, manually downloading all stuffs and 
referring to [manually_download_datasets.md](manually_download_datasets.md) for more details.**

### 4. Prepare FashionVideo dataset
* `--output_dir`: the $fashionvideo_root_dir;
* `--gpu_ids`: see above.

```Bash
python scripts/train/prepare_FashionVideo_dataset.py \
   --output_dir /p300/tpami/datasets_reproduce/fashionvideo \
   --gpu_ids 0,0,0,0,1,1,1,1
```

Modify the `--output_dir` and the `--gpu_ids`.

### 5. Prepare Place2 dataset

Place2: http://places2.csail.mit.edu/download.html

1. Manually download the Data of Places365-Standard with High-resolution images, and save them to $place_dir

* [Train images](http://data.csail.mit.edu/places/places365/train_large_places365standard.tar), 105GB. MD5: 67e186b496a84c929568076ed01a8aa1
* [Validation images](http://data.csail.mit.edu/places/places365/val_large.tar), 2.1GB. MD5: 9b71c4993ad89d2d8bcbdc4aef38042f

2. extract all images.

```shell
cd $place_dir

tar -xvf train_large_places365standard.tar
mv data_large train

tar -xvf val_large.tar
mkdir val
mv val_large val
```

### 6. Visualize all processed datasets
From the above 5 steps, we will arrive at a folder structure as follows:
```Bash
$dataset_dir

|-- places
|   |-- train
|   |   |-- a
|   |   |-- b
|   |   |-- c
|   | ......
|   |   |-- v
|   |   |-- w
|   |   |-- y
|   |   `-- z
|   |-- val
|   |   |-- val
|-- fashionvideo
|   |-- fashion_test.txt
|   |-- fashion_train.txt
|   |-- models
|   |-- primitives
|   |-- train.txt
|   |-- val.txt
|   `-- videos
|-- iPER
|   |-- images_HD
|   |-- images_HD.tar.gz
|   |-- images_HD_splits.txt
|   |-- models
|   |-- primitives
|   |-- splits
|   |-- train.txt
|   `-- val.txt
`-- motionSynthetic
    |-- poses
    |-- poses.zip
    |-- primitives
    |-- train.txt
    |-- val.txt
    |-- videos
    `-- videos.zip
```

* Open the visdom server at other terminal. Here, we use the 31102 port, and our local IP (test) is http://10.10.10.100
```Bash
python -m visdom.server -port 31102
```


* Visualize iPER processed dataset
```Bash
python scripts/train/visual_processed_data.py --gpu_ids 0 \
        --dataset_mode  ProcessedVideo \
        --dataset_dirs  $iPER_root_dir  \
        --visdom_ip http://10.10.10.100  --visdom_port 31102
```

* Visualize iPER + motionSynthetic + fashionvideo processed dataset,
modify `--dataset_dirs` and use space as the separator, e.g 
  
    `--dataset_dirs /p300/tpami/iPER /p300/tpami/motionSynthetic /p300/tpami/fashionvideo`

```Bash
python scripts/train/visual_processed_data.py --gpu_ids 0 \
        --dataset_mode  ProcessedVideo \
        --dataset_dirs  $iPER_root_dir $motionSynthetic_root_dir $fashionvideo_root_dir \
        --visdom_ip http://10.10.10.100  --visdom_port 31102
```

* Visualize iPER + Place2 processed dataset
```Bash
python scripts/train/visual_processed_data.py --gpu_ids 0 \
        --dataset_mode  ProcessedVideo+Place2 \
        --dataset_dirs  $iPER_root_dir  \
        --background_dir  $place_dir    \
        --visdom_ip http://10.10.10.100  --visdom_port 31102
```

* Visualize iPER + motionSynthetic + fashionvideo + Place2 processed dataset,
```Bash
python scripts/train/visual_processed_data.py --gpu_ids 0 \
        --dataset_mode  ProcessedVideo+Place2 \
        --dataset_dirs  $iPER_root_dir $motionSynthetic_root_dir $fashionvideo_root_dir \
        --background_dir  $place_dir  \
        --visdom_ip http://10.10.10.100  --visdom_port 31102
```

## Run training scripts
We train our model in RTX 2080Ti with 11G GPU memory.

### Train AttLWB on iPER + MotionSynthetic + FashionVideo + Place2 datasets
Modify `$iPER_root_dir`, `$motionSynthetic_root_dir`, `$fashionvideo_root_dir` and `$place_dir` with your own dataset path.

Modify `--output_dir`;

If you use single gpu, set `--gpu_ids 0`;

If it encounters with `Out of memory`, you can try to decrease `--num_source` and `--time_step`;

If you use the GPU have more than 11G GPU memory, e.g 32G TITAN-V 100, you can increase `--batch_size` or `--time_step`.

Using tensorboard for visualization, and early stop the training procedure if the training model encounters into collapse.

The default training iterations is 400000, modify it by `--Train.niters_or_epochs_no_decay`;

The default learning rate (G/D) is 0.0001, modify it by `--Train.lr_G` (`--Train.lr_D`).

More hyper-parameters could be modified in [train_aug_bg.toml](../assets/configs/trainers/train_aug_bg.toml).

```Bash
python scripts/train/dist_train.py --gpu_ids 0,1,2,3 \
        --dataset_mode    ProcessedVideo+Place2 \
        --dataset_dirs    $iPER_root_dir $motionSynthetic_root_dir $fashionvideo_root_dir \
        --background_dir  $place_dir  \
        --output_dir      $output_dir \
        --model_id   AttLWB_iPER+MS+Fashion+Place2 \
        --image_size 512 \
        --num_source 4   \
        --time_step  2   \
        --batch_size 1   --Train.niters_or_epochs_no_decay 400000
```

### Train AttLWB on iPER dataset only
```Bash
python scripts/train/dist_train.py --gpu_ids 0,1,2,3 \
        --dataset_mode    ProcessedVideo \
        --dataset_dirs    $iPER_root_dir \
        --output_dir      $output_dir \
        --model_id   AttLWB_iPER \
        --image_size 512 \
        --num_source 4   \
        --time_step  2   \
        --batch_size 1 
```

### Train AddLWB on iPER + MotionSynthetic + FashionVideo + Place2 datasets
```Bash
python scripts/train/dist_train.py --gpu_ids 0,1,2,3 \
        --dataset_mode    ProcessedVideo+Place2 \
        --dataset_dirs    $iPER_root_dir $motionSynthetic_root_dir $fashionvideo_root_dir \
        --background_dir  $place_dir  \
        --output_dir      $output_dir \
        --model_id   AddLWB_iPER+MS+Fashion+Place \
        --image_size 512 \
        --num_source 4   \
        --time_step  2   \
        --batch_size 1   \
        --gen_name   AddLWB  \
        --neural_render_cfg_path  ./assets/configs/neural_renders/AddLWB.toml
```

### Visualize the training procedure by TensorBoard
```Bash
cd   $output_dir/models/model_id

tensorboard  --logdir ./   --port 66666
```
