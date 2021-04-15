# Run Human Motion Imitation

The options to run the scripts.

- gpu_ids (str): the gpu_ids, default is "0";
- image_size (int): the image size, default is 512;
- num_source (int): the number of source images for Attention, default is 2. Large needs more GPU memory;
- assets_dir (str): the assets directory. This is very important, and there are the configurations and all pre-trained checkpoints;
- output_dir (str): the output directory;

 - src_path (str): the source input information. 
       All source paths and it supports multiple paths, uses "|" as the separator between all paths. 
       The format is "src_path_1|src_path_2|src_path_3". 
       
       Each src_input is "path?=path1,name?=name1,bg_path?=bg_path1". 
       
       It must contain 'path'. If 'name' and 'bg_path' are empty, they will be ignored.

       The 'path' could be an image path, a path of a directory contains source images, and a video path.

       The 'name' is the rename of this source input, if it is empty, we will ignore it, and use the filename of the path.

       The 'bg_path' is the actual background path if provided, otherwise we will ignore it.
       
       There are several examples of formated source paths,

        1. "path?=path1,name?=name1,bg_path?=bg_path1|path?=path2,name?=name2,bg_path?=bg_path2",
        this input will be parsed as [{path: path1, name: name1, bg_path:bg_path1},
        {path: path2, name: name2, bg_path: bg_path2}];

        2. "path?=path1,name?=name1|path?=path2,name?=name2", this input will be parsed as
        [{path: path1, name:name1}, {path: path2, name: name2}];

        3. "path?=path1", this input will be parsed as [{path: path1}].

        4. "path1", this will be parsed as [{path: path1}].

 - ref_path (str): the reference input information.
       
       All reference paths. It supports multiple paths, and uses "|" as the separator between all paths.
       The format is "ref_path_1|ref_path_2|ref_path_3".

       Each ref_path is "path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=150,effect?=View-45;View-180;BT-30-180;BT-100-800".

       It must contain 'path', and others could be empty, and they will be ignored.

       The 'path' could be an image path, a path of a directory contains images of a same person, and a video path.

       The 'name' is the rename of this source input, if it is empty, we will ignore it, and use the filename of the path.

       The 'audio' is the audio path, if it is empty, we will ignore it. If the 'path' is a video,
        you can ignore this, and we will firstly extract the audio information of this video (if it has audio channel).

       The 'fps' is fps of the final outputs, if it is empty, we will set it as the default fps 25.

       The 'pose_fc' is the smooth factor of the temporal poses. The smaller of this value, the smoother of the temporal poses. If it is empty, we will set it as the default 300. In the most cases, using the default 300 is enough, and if you find the poses of the outputs are not stable, you can decrease this value. Otherwise, if you find the poses of the outputs are over stable, you can increase this value.

       The 'cam_fc' is the smooth factor of the temporal cameras (locations in the image space). The smaller of this value, the smoother of the locations in sequences. If it is empty, we will set it as the default 150. In the most cases, the default 150 is enough;
   
       The 'effect' is the visual effect inputs. Currently, it supports `View-degree` (Novel View) and `BT-frameID-duration` (Bullet-time effect):
            --`View-degree`, e.g `View-45;View-180` means rendering the results under the global view with 45 and 180 degree in y-axis. Currently, it only supports y-axis-based novel view;
            --`BT-frameID-duration`, e.g `BT-30-180;BT-100-800` means adding two bullet-time effects. One of them is at the 30-th frame with 180 frame duration,
               and the left is at the 100-th frame with 800 frame duration.

       There are several examples of formated reference paths,

        1. "path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=150,effect?=View-45;View-180;BT-30-180;BT-100-800|
            path?=path2,name?=name2,audio?=audio_path2,fps?=25,pose_fc?=450,cam_fc?=200,effect?=View-45;BT-30-180"|
            path?=path2,name?=name2,audio?=audio_path2,fps?=25,pose_fc?=450,cam_fc?=200"|
            this input will be parsed as
            [{path: path1, name: name1, audio: audio_path1, fps: 30, pose_fc: 300, cam_fc: 150, effect: "View-45;View-180;BT-30-180;BT-100-800"},
             {path: path2, name: name2, audio: audio_path2, fps: 25, pose_fc: 450, cam_fc: 200, effect: "View-45;BT-30-180"},
             {path: path2, name: name2, audio: audio_path2, fps: 25, pose_fc: 450, cam_fc: 200}]

        2. "path?=path1,name?=name1, pose_fc?=450|path?=path2,name?=name2", this input will be parsed as
        [{path: path1, name: name1, fps: 25, pose_fc: 450, cam_fc: 150},
         {path: path2, name: name2, fps: 25, pose_fc: 300, cam_fc: 150}].

        3. "path?=path1|path?=path2", this input will be parsed as
        [{path: path1, fps:25, pose_fc: 300, cam_fc: 150}, {path: path2, fps: 25, pose_fc: 300, cam_fc: 150}].

        4. "path1|path2", this input will be parsed as
        [{path: path1, fps:25, pose_fc: 300, cam_fc: 150}, {path: path2, fps: 25, pose_fc: 300, cam_fc: 150}].

        5. "path1", this will be parsed as [{path: path1, fps: 25, pose_fc: 300, cam_fc: 150}].


## Run a single image as the source inputs
In this case, there is only a frontal body image as the source inputs.

- imitates one references
```shell

python demo/motion_imitate.py --gpu_ids 0 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "donald_trump_2" \
   --src_path   "path?=./assets/samples/sources/donald_trump_2/00000.PNG,name?=donald_trump_2" \
   --ref_path   "path?=./assets/samples/references/akun_2.mp4,name?=akun_2,pose_fc?=300"

```

- imitates two references
```shell

python demo/motion_imitate.py --gpu_ids 2 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "donald_trump_2" \
   --src_path   "path?=./assets/samples/sources/donald_trump_2/00000.PNG,name?=donald_trump_2" \
   --ref_path   "path?=./assets/samples/references/akun_2.mp4,name?=akun_2,pose_fc?=300|path?=./assets/samples/references/mabaoguo_short.mp4,name?=mabaoguo_short,pose_fc?=400"
```


## Run a folder as the source inputs
In this case, there are two source images. The one is the frontal image, and the other is the backside image.

```shell
python demo/motion_imitate.py --gpu_ids 2 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "axing_1" \
   --src_path   "path?=./assets/samples/sources/axing_1,name?=axing_1" \
   --ref_path   "path?=./assets/samples/references/bantangzhuyi_1.mp4,name?=bantangzhuyi_1,pose_fc?=300"
```


## Run with a real background as the source inputs
In this case, there are two source images. The one is the frontal image, and the other is the backside image.

```shell
python demo/motion_imitate.py --gpu_ids 2 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "afan_6=ns=2" \
   --src_path   "path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6=ns=2,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG" \
   --ref_path   "path?=./assets/samples/references/akGexYZug2Q_2.mp4.mp4,name?=akGexYZug2Q_2,pose_fc?=300"
```

## Run your own custom inputs
You can upload your own custom source images, and reference videos with the followings guidelines.

#### Source Guidelines:
 - Try to capture the source images with the same static background without too complex scene structures. If possible, we recommend using the
actual background.
 - The person in the source images holds an A-pose for introducing the most visible textures.
 - It is recommended to capture the source images in an environment without too much contrast in lighting conditions and lock auto-exposure and auto-focus of the camera.
 
#### Reference Guidelines:
  - Make sure that there is only **one** person in the referent video. Since,currently, our system does not support multiple people tracking. If there are multiple people, you need firstly use other video processing tools to crop the video.
  - Make sure that capture the video with full body person. Half body will result in bad results.
  - Try to capture the video with the static camera lens, and make sure that there is no too much zoom-in, zoom-out, panning, lens swichtings, and camera transitions. If there are multiple lens switchting and camera transitions, you need firstly use other video processing tools to crop the video.


# Run Novel View Synthesis

## Render Novel View with T-pose
```shell
python demo/novel_view.py --gpu_ids 0 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "afan_6=ns=2" \
   --src_path   "path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6=ns=2,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG" \
   --T_pose
```

## Render Novel View with the original pose
```shell
python demo/novel_view.py --gpu_ids 0 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "afan_6=ns=2" \
   --src_path   "path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6=ns=2,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG"
```

## Run Motion Imitation with Bullet-time Effect
```shell
python demo/motion_imitate.py --gpu_ids 1 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "afan_6=ns=2" \
   --src_path   "path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6=ns=2,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG" \
   --ref_path   "path?=./assets/samples/references/akun_2.mp4,name?=akun_2,pose_fc?=300,effect?=BT-30-180;BT-95-180;BT-140-180;BT-180-180;BT-220-180;BT-420-180;BT-470-180"
```
[![PaperVideo](https://img.youtube.com/vi/L7hjejkS4kE/0.jpg)](https://youtu.be/L7hjejkS4kE)

## Run Motion Imitation with Multi-View Outputs

```shell
python demo/motion_imitate.py --gpu_ids 1 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "afan_6=ns=2" \
   --src_path   "path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6=ns=2,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG" \
   --ref_path   "path?=./assets/samples/references/akun_2.mp4,name?=akun_2,pose_fc?=300,effect?=View-45;View-90;View-180;View-270"
```
[![PaperVideo](https://img.youtube.com/vi/I-3au3_i0Vc/0.jpg)](https://youtu.be/I-3au3_i0Vc)
# Run Human Appearance Transfer

Given the source images (A), source images (B), and a reference video (C), we are going to synthesize the images of A 
wearing the cloth of B and dancing like C.

All the inputs flags are similar to Human Motion Imitation and Novel View Synthesis, except for the `--src_path`.
In appearance transfer, the `--src_path` is the followings:

`--src_path` (str): the source input information. 

    All source paths and it supports multiple paths, uses "|" as the separator between all paths. 
    The format is "src_path_1|src_path_2|src_path_3".
    
    Each src_input is "path?=path1,name?=name1,bg_path?=bg_path1,parts?=part1-part2". 
    
    It must contain 'path'. If 'name' and 'bg_path' are empty, they will be ignored.
   
    The 'path' could be an image path, a path of a directory contains source images, and a video path.
   
    The 'name' is the rename of this source input, if it is empty, we will ignore it, and use the filename of the path.
   
    The 'bg_path' is the actual background path if provided, otherwise we will ignore it.
    
    The `parts?` is the selected parts of this source input. Here, we use `-` as the separator among different parts.
    The valid part names are {head, torso, left_leg, right_leg, left_arm, right_arm, left_root, right_root, 
    left_hand, right_hand, facial, upper, lower, body, all},

    {
        "head": [0],
        "torso": [1],
        "left_leg": [2],
        "right_leg": [3],
        "left_arm": [4],
        "right_arm": [5],
        "left_foot": [6],
        "right_foot": [7],
        "left_hand": [8],
        "right_hand": [9],
        "facial": [10],
        "upper": [1, 2, 3, 8, 9],
        "lower": [4, 5, 6, 7],
        "body": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    
    A whole body = {head, torso, left_leg, right_leg, left_arm, right_arm, left_root, right_root, left_hand, right_hand}.
    So, there are no over-lap regions among {head, torso, left_leg, right_leg, left_arm, right_arm, left_root, right_root, left_hand, right_hand},
    and the left parts might have over-lap regions with each other.

    Here we show some examples, "src_path_1|src_path_2", and "src_path_1" will be set as the primary source inputs.
    
    The synthesized background is come from the primary source inputs ("src_path_1").
    
    If all selected parts are not enough being a whole body = {head, torso, left_leg, right_leg, left_arm, right_arm, left_root, right_root, left_hand, right_hand}.
    The left parts will come from the primary source inputs ("src_path_1").

    1. "path?=path1,name?=name1,bg_path?=bg_path1,parts?=head|path?=path2,name?=name2,bg_path?=bg_path1,parts?=body"
    It means we take the head part from the "path1" and take the left body parts from "path2".
    See A (head) + B (body) + C (dancing pose) for more details as follows.

    2. "path?=path1,name?=name1,bg_path?=bg_path1,parts?=head-torso|path?=path2,name?=name2,bg_path?=bg_path1,parts?=left_arm-right_arm-left_hand-right_hand-left_leg-right_leg-left_foot-right_foot"
    It means we take the head and torso part from the "path1" and take the left parts from "path2".
    See A (head-torso) + B (left_arm-right_arm-left_hand-right_hand-left_leg-right_leg-left_foot-right_foot) + C (dancing pose) for more details as follows.

    3. "path?=path1,name?=name1,bg_path?=bg_path1,parts?=head-torso|path?=path2,name?=name2,bg_path?=bg_path1,parts?=left_leg-right_leg-left_foot-right_foot"
    We take {head, torso} from "path1" and {left_leg, right_leg, left_foot, right_foot} from "path2",
    and the selected parts are {head, torso, left_leg, right_leg, left_foot, right_foot}, 
    and the left parts are {left_arm, right_arm, left_hand, right_hand} will be selected from the primary source inputs ("path1").
    Therefore the actual selected parts of "path1" are {head, torso, left_arm, right_arm, left_hand, right_hand}.
    See A (head-torso) + B (left_leg-right_leg-left_foot-right_foot) + C (dancing pose) as follows.
    
    4. "path?=path1,name?=name1,bg_path?=bg_path1,parts?=head-torso|path?=path2,name?=name2,bg_path?=bg_path1,parts?=upper|path?=path3,name?=name3,parts?=lower"
    There are 3 source inputs. See A (head) + B (upper) + D (lower) + C (dancing pose) as follows for more details.

Besides, in human appearance transfer, it is better to name the `--model_id` as `name1+name2`. The reason is that in the personalization stage,
we will fine tune the model on both `name1` and `name2`, which means the personalized model is used for `name1` and `name2`. Therefore, 
it is better to rename the `--model_id` as a name neither same to `name1` nor `name2`.
 
## Human Appearance Transfer with Motion Imitation

Here we denote 
   * `donald_trump_2` as A,  actual path is `./assets/samples/sources/donald_trump_2`
   * `afan_6=ns=2` as B,  and its actual path is `./assets/samples/sources/afan_6/afan_6=ns=2`
   * `akun_1` as C, and its actual path is `./assets/samples/references/akun_1.mp4`

### A (head) + B (body) + C (dancing pose)
Here, A is the primary source inputs.

```shell
python demo/appearance_transfer.py --gpu_ids 1 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "afan_6=ns=2+trump" \
   --src_path   "path?=./assets/samples/sources/donald_trump_2/00000.PNG,name?=donald_trump_2,parts?=head|path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6=ns=2,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG,parts?=body" \
   --ref_path   "path?=./assets/samples/references/akun_1.mp4,name?=akun_1,pose_fc?=300"
```
[![PaperVideo](https://img.youtube.com/vi/6kEvJWCsfHQ/0.jpg)](https://youtu.be/6kEvJWCsfHQ)

### B (head) + A (body) + C (dancing pose)
Here, B is the primary source inputs.

```shell
python demo/appearance_transfer.py --gpu_ids 1 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "afan_6=ns=2+trump" \
   --src_path   "path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6=ns=2,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG,parts?=head|path?=./assets/samples/sources/donald_trump_2/00000.PNG,name?=donald_trump_2,parts?=body" \
   --ref_path   "path?=./assets/samples/references/akun_1.mp4,name?=akun_1,pose_fc?=300"
```
[![PaperVideo](https://img.youtube.com/vi/2jzZ7r_mmuQ/0.jpg)](https://youtu.be/2jzZ7r_mmuQ)

### A (head-torso) + B (left_arm-right_arm-left_hand-right_hand-left_leg-right_leg-left_foot-right_foot) + C (dancing pose)

```shell
python demo/appearance_transfer.py --gpu_ids 1 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "afan_6=ns=2+trump" \
   --src_path   "path?=./assets/samples/sources/donald_trump_2/00000.PNG,name?=donald_trump_2,parts?=head-torso|path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6=ns=2,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG,parts?=left_arm-right_arm-left_hand-right_hand-left_leg-right_leg-left_foot-right_foot"   \
   --ref_path   "path?=./assets/samples/references/akun_1.mp4,name?=akun_1,pose_fc?=300" 
```
[![PaperVideo](https://img.youtube.com/vi/xZgWSmqoLzQ/0.jpg)](https://youtu.be/xZgWSmqoLzQ)

### A (head-torso) + B (left_leg-right_leg-left_foot-right_foot) + C (dancing pose)

```shell
python demo/appearance_transfer.py --gpu_ids 1 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "afan_6=ns=2+trump" \
   --src_path   "path?=./assets/samples/sources/donald_trump_2/00000.PNG,name?=donald_trump_2,parts?=head-torso|path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6=ns=2,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG,parts?=left_leg-left_foot-right_leg-right_foot" \
   --ref_path   "path?=./assets/samples/references/akun_1.mp4,name?=akun_1,pose_fc?=300"
```
[![PaperVideo](https://img.youtube.com/vi/th8jcirYQjs/0.jpg)](https://youtu.be/th8jcirYQjs)


### A (head) + B (upper) + D (lower) + C (dancing pose)

More than 2 source inputs.

```shell
python demo/appearance_transfer.py --gpu_ids 1 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "afan_6=ns=2+fange_1_ns=2+trump" \
   --src_path   "path?=./assets/samples/sources/donald_trump_2/00000.PNG,name?=donald_trump_2,parts?=head|path?=./assets/samples/sources/fange_1/fange_1_ns=2,name?=fange_1_ns=2,parts?=upper|path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6=ns=2,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG,parts?=lower" \
   --ref_path   "path?=./assets/samples/references/akun_1.mp4,name?=akun_1,pose_fc?=300"
```
[![PaperVideo](https://img.youtube.com/vi/yp08eVzA0BY/0.jpg)](https://youtu.be/yp08eVzA0BY)

## Human Appearance Transfer with Motion Imitation and Novel View Synthesis


### Multi-view outputs of Appearance Transfer and Motion Imitation

```shell
python demo/appearance_transfer.py --gpu_ids 1 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "afan_6=ns=2+trump" \
   --src_path   "path?=./assets/samples/sources/donald_trump_2/00000.PNG,name?=donald_trump_2,parts?=head|path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6=ns=2,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG,parts?=body" \
   --ref_path   "path?=./assets/samples/references/akun_2.mp4,name?=akun_2,pose_fc?=300,effect?=View-0;View-90;View-180;View-270"
```
[![PaperVideo](https://img.youtube.com/vi/ND79BNwiEro/0.jpg)](https://youtu.be/ND79BNwiEro)

### Bullet-time effects on Appearance Transfer and Motion Imitation

```shell
python demo/appearance_transfer.py --gpu_ids 1 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "afan_6=ns=2+trump" \
   --src_path   "path?=./assets/samples/sources/donald_trump_2/00000.PNG,name?=donald_trump_2,parts?=head|path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6=ns=2,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG,parts?=body" \
   --ref_path   "path?=./assets/samples/references/akun_2.mp4,name?=akun_2,pose_fc?=300,effect?=BT-30-180;BT-95-180;BT-140-180;BT-180-180;BT-220-180;BT-420-180;BT-470-180|path?=./assets/samples/references/Fortnite_orange_justice.mp4,name?=Fortnite_orange_justice,pose_fc?=350,effect?=BT-30-180;BT-130-180;BT-240-180;BT-350-180;BT-460-180"
```
| Demo 1                                                                                      | Demo 2                                                                                                             |
|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [![PaperVideo](https://img.youtube.com/vi/_oW1Ir1PCRE/0.jpg)](https://youtu.be/_oW1Ir1PCRE) | [![QualitativeResults](https://img.youtube.com/vi/xKHSC4i4RFc/0.jpg)](https://www.youtube.com/watch?v=xKHSC4i4RFc) |