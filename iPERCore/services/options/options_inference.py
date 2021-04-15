# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

from .options_base import BaseOptions


class InferenceOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        src_input_desc = """
            All source paths and it supports multiple paths, uses "|" as the separator between all paths. 
            The format is "src_path_1|src_path_2|src_path_3". 
            Each src_input is "path?=path1,name?=name1,bg_path?=bg_path1". 
            It must contain 'path'. If 'name' and 'bg_path' are empty, they will be ignored.
            
            The 'path' could be an image path, a path of a directory contains source images, and a video path.
            
            The 'name' is the rename of this source input, if it is empty, we will ignore it, 
            and use the filename of the path.
            
            The 'bg_path' is the actual background path if provided, otherwise we will ignore it.
            
            There are several examples of formated source paths,
            
            1. "path?=path1,name?=name1,bg_path?=bg_path1|path?=path2,name?=name2,bg_path?=bg_path2", 
            this input will be parsed as [{path: path1, name: name1, bg_path:bg_path1}, 
            {path: path2, name: name2, bg_path: bg_path2}];
            
            2. "path?=path1,name?=name1|path?=path2,name?=name2", this input will be parsed as 
            [{path: path1, name:name1}, {path: path2, name: name2}];
            
            3. "path?=path1", this input will be parsed as [{path: path1}]. 
            
            4. "path1", this will be parsed as [{path: path1}].
        """

        self._parser.add_argument("--src_path", type=str, required=True, help=src_input_desc)

        ref_input_desc = """
            All reference paths. It supports multiple paths, and uses "|" as the separator between all paths. 
            The format is "ref_path_1|ref_path_2|ref_path_3". 
            Each ref_path is "path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=400,cam_fc?=150".
            It must contain 'path', and others could be empty, and they will be ignored.
            
            The 'path' could be an image path, a path of a directory contains source images, and a video path.
            
            The 'name' is the rename of this source input, if it is empty, we will ignore it,
            and use the filename of the path.
            
            The 'audio' is the audio path, if it is empty, we will ignore it.
            
            The 'fps' is fps of the final outputs, if it is empty, we will set it as the default fps 25.
            
            The 'pose_fc' is the smooth factor of the temporal poses. The smaller of this value, the smoother of the 
            temporal poses. If it is empty, we will set it as the default 400. In the most cases, using the default
            400 is enough, and if you find the poses of the outputs are not stable, you can decrease this value.
            Otherwise, if you find the poses of the outputs are over stable, you can increase this value.
            
            The 'cam_fc' is the smooth factor of the temporal cameras (locations in the image space). The smaller of
            this value, the smoother of the locations in sequences. If it is empty, we will set it as the default 150.
            In the most cases, the default 150 is enough.
            
            1. "path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=400,cam_fc?=150|
                path?=path2,name?=name2,audio?=audio_path2,fps?=25,pose_fc?=450,cam_fc?=200", 
                this input will be parsed as 
                [{path: path1, name: name1, audio: audio_path1, fps: 30, pose_fc: 400, cam_fc: 150},
                 {path: path2, name: name2, audio: audio_path2, fps: 25, pose_fc: 450, cam_fc: 200}]
            
            2. "path?=path1,name?=name1, pose_fc?=450|path?=path2,name?=name2", this input will be parsed as 
            [{path: path1, name: name1, fps: 25, pose_fc: 450, cam_fc: 150}, 
             {path: path2, name: name2, fps: 25, pose_fc: 400, cam_fc: 150}]. 
            
            3. "path?=path1|path?=path2", this input will be parsed as 
            [{path: path1, fps:25, pose_fc: 400, cam_fc: 150}, {path: path2, fps: 25, pose_fc: 400, cam_fc: 150}].
            
            4. "path1|path2", this input will be parsed as
            [{path: path1, fps:25, pose_fc: 400, cam_fc: 150}, {path: path2, fps: 25, pose_fc: 400, cam_fc: 150}].
            
            5. "path1", this will be parsed as [{path: path1, fps: 25, pose_fc: 400, cam_fc: 150}].

        """

        self._parser.add_argument("--ref_path", type=str, help=ref_input_desc)

        self._parser.add_argument("--has_personalize", action="store_true", help="has personalization or not.")

        self._parser.add_argument("--T_pose", action="store_true", default=False,
                                  help="view as T pose or not in human novel view synthesis.")

        # visualizer
        self._parser.add_argument("--ip", type=str, default="", help="visdom ip")
        self._parser.add_argument("--port", type=int, default=31150, help="visdom port")

        self.is_train = False
