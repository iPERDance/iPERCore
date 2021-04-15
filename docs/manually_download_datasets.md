# Manually download iPER and MotionSynthetic datasets

All available download links:

* [OneDrive 1](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/liandz_shanghaitech_edu_cn/ErkIzzi6n0RLrP9gP5k2tpcB2BRzeRMok9moOgEUnpqX8A?e=Pq1omh)

* [OneDrive 2](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/liuwen_shanghaitech_edu_cn/EiOrbTo4yUtBrgQ4KiKswxUB-UyYl69W-pSVMNeFcXwYIw?e=z1Fyea)

* [OneDrive 3](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/qianych_shanghaitech_edu_cn/Evg7YNYDV5xGjox7qIsUe1IBvh3vodNPY1-1x4JpfX1bcQ?e=oW6Qcn)

* [BaiduPan](https://pan.baidu.com/s/1zEpPaU505Df13LOyF-H3Pw), password: `uomm` 

## Download iPER dataset

In the local disk, the iPER root folder is $iPER_root_dir, and `cd` to $iPER_root_dir.

1. download `train.txt`, `val.txt`, and move them into $iPER_root_dir
2. download all split files from `splits/images_HD.tar.gz.aa` to `splits/images_HD.tar.gz.bk`, and 
   move them to $iPER_root_dir/splits, and we will arrive at the following file structure:
    ```shell
    $iPER_root_dir:
        |-- splits
        |   |-- images_HD.tar.gz.aa
        |   |-- images_HD.tar.gz.ab
        |   |-- images_HD.tar.gz.ac
        |   |-- ......
        |   |-- images_HD.tar.gz.bj
        |   `-- images_HD.tar.gz.bk
        |-- train.txt
        `-- val.txt
    ```

3. merge all the `splits` files into `images_HD.tar.gz` and move it to $iPER_root_dir,
    ```Bash
    cat splits/images_HD.tar.gz.*   >   images_HD.tar.gz
    ```
    Then, we will arrive at the following file structure
    
    ```shell
    $iPER_root_dir:
        |-- splits
        |   |-- images_HD.tar.gz.aa
        |   |-- images_HD.tar.gz.ab
        |   |-- images_HD.tar.gz.ac
        |   |-- ......
        |   |-- images_HD.tar.gz.bj
        |   `-- images_HD.tar.gz.bk
        |-- images_HD.tar.gz
        |-- train.txt
        `-- val.txt
    ```

4. extract `images_HD.tar.gz` to $iPER_root_dir,
   ```Bash
   tar  -xzvf images.tar.gz
   ```
   Then, we will arrive at the following file structure
    ```shell
    $iPER_root_dir:
        |-- images_HD
        |   |-- 001
        |   |   |-- 1
        |   |   |   |-- 1
        |   |   |   `-- 2
        |   |   |-- 10
        |   |   |   |-- 1
        |   |   |   `-- 2
        |   |   |-- 11
        |   |   |   |-- 1
        |   |   |   `-- 2
        |   |   |-- 12
        |  .......
        |   |-- 029
        |   |   `-- 1
        |   |   |   |-- 1
        |   |   |   `-- 2
        |   `-- 030
        |       `-- 1
        |   |   |   |-- 1
        |   |   |   `-- 2
        |-- splits
        |   |-- images_HD.tar.gz.aa
        |   |-- images_HD.tar.gz.ab
        |   |-- images_HD.tar.gz.ac
        |   |-- ......
        |   |-- images_HD.tar.gz.bj
        |   `-- images_HD.tar.gz.bk
        |-- images_HD.tar.gz
        |-- train.txt
        `-- val.txt
    ```

5. run the following script to process the iPER dataset
   ```Bash
   python scripts/train/prepare_iPER_dataset.py \
      --output_dir $iPER_root_dir \
      --gpu_ids 0,0,0,1,1,1,2,2,2,3,3,3,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9
   ```
   
   Then, all the processed information will be saved into $iPER_root_dir/primitives, and we will arrive at the 
   following file structure,
   ```Bash
    $iPER_root_dir:
     |-- images_HD
     |   |-- 001
     |   |   |-- 1
     |   |   |   |-- 1
     |   |   |   `-- 2
     |   |   |-- 10
     |   |   |   |-- 1
     |   |   |   `-- 2
     |   |   |-- 11
     |   |   |   |-- 1
     |   |   |   `-- 2
     |   |   |-- 12
     |  .......
     |   |-- 029
     |   |   `-- 1
     |   |   |   |-- 1
     |   |   |   `-- 2
     |   `-- 030
     |       `-- 1
     |   |   |   |-- 1
     |   |   |   `-- 2
     |-- splits
     |   |-- images_HD.tar.gz.aa
     |   |-- images_HD.tar.gz.ab
     |   |-- images_HD.tar.gz.ac
     |   |-- ......
     |   |-- images_HD.tar.gz.bj
     |   `-- images_HD.tar.gz.bk
     |-- images_HD.tar.gz
     |-- primitives
     |   |-- 001
     |   |   |-- 1
     |   |   |   |-- 1
     |   |   |   `-- 2
     |   |   |-- 10
     |   |   |   |-- 1
     |   |   |   `-- 2
     |   |-- 005
     |   |   `-- 1
     |   |       |-- 1
     |   |       `-- 2
     |  ......
     |   `-- 030
     |       `-- 1
     |           |-- 1
     |           `-- 2
     |-- train.txt
     `-- val.txt
   ```

## Download motionSynthetic dataset
The MotionSynthetic root folder is $MotionSynthetic_root_dir, and `cd` to $MotionSynthetic_root_dir.

1. download `train.txt` and `val.txt`, and move them to $MotionSynthetic_root_dir;

2. download `videos.zip` and `poses.zip`, and move them to $MotionSynthetic_root_dir;

3. unzip `videos.zip` and `poses.zip`
   ```Bash
   unzip videos.zip
   unzip poses.zip
   ```
   Then, we will arrive at the following file structure:
   ```Bash
   $MotionSynthetic_root_dir
   |-- poses
   |   |-- MG_125611487366942_0366_0018
   |   |   |-- kps.pkl
   |   |   `-- pose_shape.pkl
   |   |-- MG_125611494277906_2031_0017
   |   |   |-- kps.pkl
   |   |   `-- pose_shape.pkl
   |   | ......
   |   `-- PeopleSnapshot_male-9-plaza_2211_0016
   |       |-- kps.pkl
   |       `-- pose_shape.pkl
   |-- poses.zip
   |-- train.txt
   |-- val.txt
   |-- videos
   |   |-- MG_125611487366942_0366_0018.mp4
   | ......
   |   `-- PeopleSnapshot_male-9-plaza_2211_0016.mp4
   `-- videos.zip

   ```

4. run the following script to process the MotionSynthetic dataset
   ```Bash
   python scripts/train/prepare_motionSynthetic_dataset.py \
      --output_dir /p300/tpami/datasets_reproduce/motionSynthetic \
      --gpu_ids 0,0,0,0,1,1,1,1
   ```
   
   Then, all the processed information will be saved into $MotionSynthetic_root_dir/primitives, 
   and we will arrive at the following file structure:
   ```Bash
   $MotionSynthetic_root_dir
   |-- poses
   |   |-- MG_125611487366942_0366_0018
   |   |   |-- kps.pkl
   |   |   `-- pose_shape.pkl
   |   |-- MG_125611494277906_2031_0017
   |   |   |-- kps.pkl
   |   |   `-- pose_shape.pkl
   |   | ......
   |   `-- PeopleSnapshot_male-9-plaza_2211_0016
   |       |-- kps.pkl
   |       `-- pose_shape.pkl
   |-- poses.zip
   |-- primitives
   |   |-- MG_125611487366942_0366_0018
   |   |   `-- processed
   |   |       |-- actual_background
   |   |       |-- background
   |   |       |-- images
   |   |       |-- parse
   |   |       |-- vid_info.pkl
   |   |       `-- visual.mp4
   |   | ......
   |   `-- PeopleSnapshot_male-9-plaza_2211_0016
   |       `-- processed
   |           |-- actual_background
   |           |-- background
   |           |-- images
   |           |-- parse
   |           |-- vid_info.pkl
   |           `-- visual.mp4
   |-- train.txt
   |-- val.txt
   |-- videos
   |   |-- MG_125611487366942_0366_0018.mp4
   | ......
   |   `-- PeopleSnapshot_male-9-plaza_2211_0016.mp4
   `-- videos.zip
   ```