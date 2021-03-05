# Training Details

## Prepare iPER dataset

### 1. Download the original video data:

    * download the `iPER_1024_video_release.zip`, and move them into $iPER_root_dir
    
    * download the `train.txt`, and move them into $iPER_root_dir
    
    * download the `val.txt`, and move them into $iPER_root_dir
    
    * unzip `$iPER_root_dir/iPER_1024_video_release.zip` into $iPER_root_dir, 
      The file structure of $iPER_root_dir will be:

   $iPER_root_dir:
        --iPER_1024_video_release.zip
        --iPER_1024_video_release
            --001_1_1.mp4
            --001_1_2.mp4
            --001_2_1.mp4
            ...
            --030_1_2.mp4
        --train.txt
        --val.txt


### 2. Preprocess:

Preprocessing all videos in $iPER_root_dir/iPER_1024_video_release.

* `--output_dir`: the iPER_root_dir;
* `--gpu_ids`: the gpu ids. The preprocessor needs at least `3.2G` GPU memory.
If we have a RTX 2080Ti with `11G`, then each GPU could run 3 preprocessors. For example,
  we have two RTX 2080Ti(s), and we can set this as `--gpu_ids 0,0,0,1,1,1` with 6 parallel preprocessors.
  
```Bash
python scripts/train/prepare_iPER_dataset.py \
   --output_dir /p300/tpami/datasets/iPER \
   --gpu_ids 0,0,0,1,1,1,2,2,2,3,3,3,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9
```


### 3. Prepare FashionVideo dataset

1. Download the original video data;

2. Preprocessing:

```Bash
python scripts/train/prepare_FashionVideo_dataset.py \
   --output_dir /p300/tpami/datasets/fashionvideo \
   --gpu_ids 0,0,0,1,1,1,2,2,2,3,3,3
```

### 4.Prepare MotionSynthetic dataset


```Bash
python scripts/train/prepare_motionSynthetic_dataset.py \
   --output_dir /p300/tpami/datasets/motionSynthetic \
   --gpu_ids 0,0,1,1,2,2,3,3
```

### 5. Prepare Place2 dataset


## Train iPER + MotionSynthetic + FashionVideo + Place2
```Bash

python 

```
