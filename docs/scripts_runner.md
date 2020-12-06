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

       Each ref_path is "path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=150".

       It must contain 'path', and others could be empty, and they will be ignored.

       The 'path' could be an image path, a path of a directory contains images of a same person, and a video path.

       The 'name' is the rename of this source input, if it is empty, we will ignore it, and use the filename of the path.

       The 'audio' is the audio path, if it is empty, we will ignore it. If the 'path' is a video,
        you can ignore this, and we will firstly extract the audio information of this video (if it has audio channel).

       The 'fps' is fps of the final outputs, if it is empty, we will set it as the default fps 25.

       The 'pose_fc' is the smooth factor of the temporal poses. The smaller of this value, the smoother of the temporal poses. If it is empty, we will set it as the default 300. In the most cases, using the default 300 is enough, and if you find the poses of the outputs are not stable, you can decrease this value. Otherwise, if you find the poses of the outputs are over stable, you can increase this value.

       The 'cam_fc' is the smooth factor of the temporal cameras (locations in the image space). The smaller of this value, the smoother of the locations in sequences. If it is empty, we will set it as the default 150. In the most cases, the default 150 is enough.

       There are several examples of formated reference paths,

        1. "path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=150|
            path?=path2,name?=name2,audio?=audio_path2,fps?=25,pose_fc?=450,cam_fc?=200",
            this input will be parsed as
            [{path: path1, name: name1, audio: audio_path1, fps: 30, pose_fc: 300, cam_fc: 150},
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

python demo/motion_imitate.py --gpu_ids 2 \
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