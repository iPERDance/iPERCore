# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import warnings
import os

from iPERCore.tools.utils.filesio.persistence import mkdir, load_pickle_file
from iPERCore.tools.utils.multimedia.video import extract_audio_from_video, get_video_fps
from iPERCore.tools.utils.multimedia.mediafiles import is_video_file

from .process_info import ProcessInfo


class MetaInputInfo(object):

    def __init__(self, path="", bg_path="", name=""):
        self.path = path
        self.bg_path = bg_path
        self.name = name

    def parse(self, input_str):
        pass

    def get_info(self):
        return self.__dict__

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, item):
        return self.__dict__[item]


class SrcMetaInputInfo(MetaInputInfo):
    """
        The meta information for the source inputs.
    """

    META_KEY_TO_TYPE = {
        "path": str,
        "bg_path": str,
        "name": str
    }

    def __init__(self, path="", bg_path="", name=""):
        """

        Args:
            path (str): the path of the video/image/the folder of the images;
            bg_path (str): the background image path;
            name (str): the rename of the input.
        """

        super().__init__(path, bg_path, name)

    def parse(self, src_str):
        """

        Parse the src_str to structure meta information,
             1. 'path?=path1,bg_path?=bg_path1'
             2. 'path1,bg_path?=bg_path1' is valid.
             3. 'path1,bg_path1,fps' is not valid, it will ignore name, fps, since we do not know its key.
        Args:
            src_str (str): e.g. 'path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=100'

        Returns:
            None
        """

        if "," in src_str:
            # use ',' to split each 'path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=100'
            # we get ['path?=path1', 'name?=name1', 'audio?=audio_path1', 'fps?=30', 'pose_fc?=300', 'cam_fc?=100']
            kv_pair_list = src_str.split(",")

            for i, kv_pair in enumerate(kv_pair_list):

                if "?=" in kv_pair:
                    key, value = kv_pair.split("?=")

                    if key in self.META_KEY_TO_TYPE and value:
                        self.__setattr__(key, self.META_KEY_TO_TYPE[key](value))
                    else:
                        warnings.warn(f"{kv_pair} is missing the value, and we will ignore it.")

                elif i == 0:
                    warnings.warn(f"{kv_pair} has no key, and it is the first item, "
                                  f"here we will set it as the `path`.")

                    self.path = kv_pair

                else:
                    warnings.warn(f"{kv_pair} is ambiguous, and we do not know its key.")

        else:
            self.path = src_str

    def __str__(self):
        _str = ""
        for key, val in self.__dict__.items():
            _str += f"{key}: {val}\n"

        return _str

    def __eq__(self, other):
        return self.path == other.path and self.bg_path == other.bg_path


class RefMetaInputInfo(MetaInputInfo):
    """
        The meta information for the reference inputs.
    """

    META_KEY_TO_TYPE = {
        "path": str,
        "name": str,
        "audio": str,
        "fps": float,
        "pose_fc": float,
        "cam_fc": float
    }

    def __init__(self, path="", name="", audio="", fps=25, pose_fc=300, cam_fc=100):
        """

        Args:
            path (str): the path of the video/image/the folder of the images;
            name (str): the primitive name for this input, if it is None, then, we the use os.path.split(path)[-1] as
                the primitive name;
            audio (str): the audio path;
            fps (float): the fps;
            pose_fc (float): the pose smooth factor, default is 300;
            cam_fc (float): the camera smooth factor, default is 100.
        """
        super().__init__(path, bg_path="")

        self.name = name
        self.audio = audio
        self.fps = fps
        self.pose_fc = pose_fc
        self.cam_fc = cam_fc

    def parse(self, ref_str):
        """

        Parse the ref_str to structure meta information,
             1. 'path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=100'
             2. 'path1,fps?=25' is valid.
             3. 'path1,name,fps' is not valid, it will ignore name, fps, since we do not know its key.
        Args:
            ref_str (str): e.g. 'path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=100'

        Returns:
            None
        """

        if "," in ref_str:
            # use ',' to split each 'path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=100'
            # we get ['path?=path1', 'name?=name1', 'audio?=audio_path1', 'fps?=30', 'pose_fc?=300', 'cam_fc?=100']
            kv_pair_list = ref_str.split(",")

            for i, kv_pair in enumerate(kv_pair_list):

                if "?=" in kv_pair:
                    key_value = kv_pair.split("?=")

                    key, value = key_value
                    if key in self.META_KEY_TO_TYPE and value:
                        self.__setattr__(key, self.META_KEY_TO_TYPE[key](value))
                    else:
                        warnings.warn(f"{kv_pair} is missing the value, and we will ignore it.")

                elif i == 0:
                    warnings.warn(f"{kv_pair} has no key, and it is the first item, "
                                  f"here we will set it as the `path`.")

                    self.path = kv_pair

                else:
                    warnings.warn(f"{kv_pair} is ambiguous, and we do not know its key. We will ignore it.")

        else:
            self.path = ref_str

        if not self.name:
            self.name = os.path.split(self.path)[-1]

    def __str__(self):
        _str = ""
        for key, val in self.__dict__.items():
            _str += f"{key}: {val}\n"

        return _str

    def __eq__(self, other):
        flag = self.path == other.path and self.name == other.name and self.fps == other.fps \
               and self.pose_fc == other.pose_fc and self.cam_fc == other.cam_fc

        return flag


class MetaProcess(object):

    def __init__(self, meta_input: MetaInputInfo, root_primitives_dir: str):
        """

        Args:
            meta_input (MetaInputInfo):
            root_primitives_dir:
        """

        primitives_dir = mkdir(os.path.join(root_primitives_dir, meta_input.name))
        processed_dir = mkdir(os.path.join(primitives_dir, "processed"))

        meta_input = self.update_meta_input(meta_input, processed_dir)

        self.meta_input = meta_input.get_info()
        self.primitives_dir = primitives_dir
        self.processed_dir = processed_dir
        self.vid_info_path = os.path.join(processed_dir, "vid_info.pkl")

    @staticmethod
    def update_meta_input(meta_input, processed_dir):
        path = meta_input["path"]

        if is_video_file(path):
            audio_path = os.path.join(processed_dir, "audio.mp3")

            if not os.path.exists(audio_path):
                extract_audio_from_video(path, audio_path)

            fps = get_video_fps(path, ret_type="float")

            meta_input["audio"] = audio_path
            meta_input.fps = fps

        return meta_input

    def get_info(self):
        return self.__dict__

    def __getitem__(self, item):
        return self.__dict__[item]

    def __str__(self):
        _str = "----------------------MetaProcess----------------------\n"
        _str += "meta_input:\n"

        for meta_input_key, meta_input_val in self.meta_input.items():
            _str += f"\t{meta_input_key}: {meta_input_val}\n"

        _str += f"primitives_dir: {self.primitives_dir}\n"
        _str += f"processed_dir: {self.processed_dir}\n"
        _str += f"vid_info_path: {self.vid_info_path}\n"

        _str += "-------------------------------------------------------"

        return _str

    def check_has_been_processed(self, verbose=True):
        # 1. check the vid_info_path exist.
        flag = os.path.exists(self.vid_info_path)

        if flag:
            context = load_pickle_file(self.vid_info_path)
            has_finished = ProcessInfo.check_has_been_processed(context, verbose)
            flag = flag and has_finished

        return flag


class MetaOutput(object):
    def __init__(self, meta_src, meta_ref):
        """

        Args:
            meta_src (MetaProcess):
            meta_ref (MetaProcess):
        """

        src_name = meta_src.meta_input["name"]
        ref_name = meta_ref.meta_input["name"]

        synthesized_dir = mkdir(os.path.join(meta_src.primitives_dir, "synthesis"))

        src_ref_name = f"{src_name}-{ref_name}"
        imitation_dir = mkdir(os.path.join(synthesized_dir, "imitations", f"{src_ref_name}"))
        swapper_dir = mkdir(os.path.join(synthesized_dir, "swappers", f"{src_ref_name}"))
        viewer_dir = mkdir(os.path.join(synthesized_dir, "viewers"))

        self.src_name = src_name
        self.ref_name = ref_name

        self.audio = meta_ref.meta_input["audio"]
        self.fps = meta_ref.meta_input["fps"]
        self.pose_fc = meta_ref.meta_input["pose_fc"]
        self.cam_fc = meta_ref.meta_input["cam_fc"]

        self.src_ref_name = src_ref_name
        self.synthesized_dir = synthesized_dir

        self.imitation_dir = imitation_dir
        self.imitation_mp4 = imitation_dir + ".mp4"

        self.swapper_dir = swapper_dir
        self.swapper_mp4 = swapper_dir + ".mp4"

        self.viewer_dir = viewer_dir
        self.viewer_mp4 = viewer_dir + ".mp4"

    def print_full_infos(self):
        _str = ""

        for key, val in self.__dict__.items():
            _str += f"{key}: {val}\n"

        return _str

    def __str__(self):
        _str = "----------------------MetaOutput----------------------\n"

        if os.path.exists(self.imitation_mp4):
            _str += f"{self.src_name} imitates {self.ref_name} in {self.imitation_mp4}\n"

        else:
            _str += f"{self.src_name} imitates {self.ref_name} failed."

        if os.path.exists(self.swapper_mp4):
            _str += f"{self.src_name} and {self.ref_name} swap appearances in {self.swapper_mp4}\n"

        if os.path.exists(self.viewer_mp4):
            _str += f"{self.src_name} has novel views in {self.viewer_mp4}\n"

        _str += "------------------------------------------------------"

        return _str

    def __repr__(self):
        return self.__str__()


def parse_ref_input(ref_input):

    """
    The reference paths, it support multiple paths, use '|' as the separator between all paths, the format is
            'tgt_input_1|tgt_input_2|tgt_input_3'. Each tgt_input is 'key1?=value1,key2?=value2,key3?=value3'.

            1. 'path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=100|
                path?=path2,name?=name2,audio?=audio_path2,fps?=25,pose_fc?=450,cam_fc?=200', this input will be
                parsed as [{path: path1, name: name1, audio: audio_path1, fps: 30, pose_fc: 300, cam_fc: 100},
                           {path: path2, name: name2, audio: audio_path2, fps: 25, pose_fc: 450, cam_fc: 200}]

            2. 'path1|path2', this input will be parsed as [{path: path1}, {path: path2}].

    Args:
        ref_input (str):

    Returns:
        ref_meta_list (list[RefMetaInfo]): each element contains the followings,
            RefMetaInfo.path (str): the path of the video/image/the folder of the images;
            RefMetaInfo.name (str): the primitive name for this input, if it is None, then, we the use
                os.path.split(path)[-1] as the primitive name;
            RefMetaInfo.audio (str): the audio path;
            RefMetaInfo.fps (float): the fps;
            RefMetaInfo.pose_fc (float): the pose smooth factor, default is 300;
            RefMetaInfo.cam_fc (float): the camera smooth factor, default is 100.
    """

    ref_meta_list = []

    # take 'path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=100|
    #       path?=path2,name?=name2,audio?=audio_path2,fps?=25,pose_fc?=450,cam_fc?=200' as an example.

    # 1. use separator '|' to split each inputs.
    # Then, we will get ['path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=100',
    #                    'path?=path2,name?=name2,audio?=audio_path2,fps?=25,pose_fc?=450,cam_fc?=200']
    ref_str_list = ref_input.split("|")
    for each_ref_str in ref_str_list:
        # use separator
        # ',' to split each 'path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=100'
        # we get ['path?=path1', 'name?=name1', 'audio?=audio_path1', 'fps?=30', 'pose_fc?=300', 'cam_fc?=100']

        if each_ref_str:
            ref_meta = RefMetaInputInfo()
            ref_meta.parse(each_ref_str)

            ref_meta_list.append(ref_meta)

    return ref_meta_list


def parse_src_input(src_input):
    """

    Args:
        src_input:

    Returns:

    """

    src_meta_list = []

    ref_str_list = src_input.split("|")
    for each_src_str in ref_str_list:
        if each_src_str:
            src_meta = SrcMetaInputInfo()
            src_meta.parse(each_src_str)

            src_meta_list.append(src_meta)

    return src_meta_list
