# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import torch
import numpy as np
import warnings
from multiprocessing import Queue, Process

from iPERCore.tools.processors.preprocessors import Preprocessor
from iPERCore.tools.human_digitalizer import deformers
from iPERCore.tools.utils.filesio.cv_utils import load_parse

from iPERCore.services.options.process_info import ProcessInfo


class PreprocessConsumer(Process):
    """
    Consumer for preprocessing, it contains the following steps:
    1. It firstly use the human detector to crop the bounding boxes of the person;
    2. then, it center crops a square image from the original image, and it might use pad and resize;
    3. next, it will estimate the 3D cam, pose, and shape of the 3D parametric model (SMPL);
    4. then, it will sort the images by counting the number of front visible triangulated faces;
    5. finally, it will run the human matting algorithm to get the mask of human;

    """

    def __init__(self, queue, gpu_id, opt):
        self.queue = queue
        self.gpu_id = gpu_id
        self.opt = opt
        self.is_run = True

        Process.__init__(self, name="PreprocessConsumer_{}".format(gpu_id))

    def run(self) -> None:
        os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        device = torch.device("cuda:0")

        processor = Preprocessor(
            cfg=self.opt,
            proc_size=self.opt.image_size,
            device=device
        )

        while self.is_run and not self.queue.empty():
            try:
                meta_proc, is_ref = self.queue.get()

                processed_info = ProcessInfo(meta_proc)
                processed_info.deserialize()

                if is_ref:
                    processor.execute(
                        processed_info,
                        crop_size=self.opt.image_size,
                        num_workers=self.opt.num_workers,
                        estimate_boxes_first=self.opt.Preprocess.estimate_boxes_first,
                        factor=self.opt.Preprocess.Cropper.ref_crop_factor,
                        use_simplify=self.opt.Preprocess.use_smplify,
                        temporal=True, filter_invalid=True, inpaintor=False,
                        parser=False, find_front=False, visual=False
                    )
                else:
                    processor.execute(
                        processed_info,
                        crop_size=self.opt.image_size,
                        num_workers=self.opt.num_workers,
                        estimate_boxes_first=self.opt.Preprocess.estimate_boxes_first,
                        factor=self.opt.Preprocess.Cropper.src_crop_factor,
                        use_simplify=self.opt.Preprocess.use_smplify,
                        temporal=False, filter_invalid=True, find_front=True, parser=True,
                        num_candidate=self.opt.Preprocess.FrontInfo.NUM_CANDIDATE,
                        render_size=self.opt.Preprocess.FrontInfo.RENDER_SIZE,
                        inpaintor=True,
                        dilate_kernel_size=self.opt.Preprocess.BackgroundInpaintor.dilate_kernel_size,
                        dilate_iter_num=self.opt.Preprocess.BackgroundInpaintor.dilate_iter_num,
                        bg_replace=self.opt.Preprocess.BackgroundInpaintor.bg_replace,
                        visual=True,
                    )

            except Exception("model error!") as e:
                print(e.message)

    def terminate(self):
        self.is_run = False


class HumanDigitalDeformConsumer(Process):
    """
        The human digital deformer. Since the 3D parametric model only has the nude body 3D mesh, and this class is
    going to estimate the 3D mesh with more details.
    """

    def __init__(self, queue, gpu_id, opt):
        self.queue = queue
        self.gpu_id = gpu_id
        self.opt = opt
        self.is_run = True

        Process.__init__(self, name="HumanDigitalDeformConsumer_{}".format(gpu_id))

    @staticmethod
    def check_has_been_deformed(processed_vid_info):
        """

        Args:
            processed_vid_info:

        Returns:

        """

        has_run_deform = processed_vid_info["has_run_deform"]

        return has_run_deform

    def run(self) -> None:
        os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        digital_type = self.opt.digital_type

        if digital_type == "cloth_smpl_link":
            deformer = deformers.ClothSmplLinkDeformer(
                cloth_parse_ckpt_path=self.opt.Preprocess.Deformer.cloth_parse_ckpt_path,
                smpl_model=self.opt.smpl_model,
                part_path=self.opt.part_path
            )
        else:
            deformer = None

        while self.is_run and not self.queue.empty():
            try:
                process_info = self.queue.get()

                if self.opt.digital_type == "cloth_smpl_link":
                    prepared_inputs = self.prepare_inputs_for_run_cloth_smpl_links(process_info)
                    has_links, links = deformer.find_links(**prepared_inputs)
                    if has_links:
                        process_info["processed_deform"]["links"] = links

                elif self.opt.digital_type == "sil2smpl":
                    prepared_inputs = self.prepare_inputs_for_run_sil2smpl_offsets(process_info)
                    offsets = deformers.run_sil2smpl_offsets(**prepared_inputs)

                    process_info["processed_deform"]["offsets"] = offsets

                else:
                    warnings.warn(f"there is no digital type of {self.opt.digital_type}")

                process_info["has_run_deform"] = True
                process_info.serialize()

            except Exception("model error!") as e:
                print(e.message)

    def terminate(self):
        self.is_run = False

    def prepare_inputs_for_run_sil2smpl_offsets(self, process_info):
        """
            Prepare the inputs for run_sil2smpl_offsets. It will return a dict, contains the
        keys of img_path, output_dir, init_smpls;

        Args:
            process_info (ProcessInfo):

        Returns:
            prepared_info (dict): the prepared information, which contains:
                --obs_sils (np.ndarray): (number of source, 1, image_size, image_size) is in the range [0, 1];
                --init_smpls (np.ndarray): (number of source, 85).
        """

        # 1. load the processed information by PreprocessConsumer
        src_infos = process_info.convert_to_src_info(self.opt.num_source)

        src_ids = src_infos["src_ids"]
        src_smpls = src_infos["smpls"][src_ids]
        alpha_paths = src_infos["alpha_paths"]

        masks = []

        for i in src_ids:
            parse_path = alpha_paths[i]
            mask = load_parse(parse_path, self.opt.image_size)

            masks.append(mask)

        prepared_info = {
            "obs_sils": np.stack(masks, axis=0),
            "init_smpls": src_smpls,
        }

        return prepared_info

    def prepare_inputs_for_run_cloth_smpl_links(self, process_info):
        """
            Prepare the inputs for human_digitalizer.ClothSmplLinkDeformer().find_links().
            It will return a dict, contains the keys of img_path, output_dir, init_smpls;

        Args:
            process_info (ProcessInfo):

        Returns:
            prepared_info (dict): the prepared information, which contains:
                --obs_sils (np.ndarray): (number of source, 1, image_size, image_size) is in the range [0, 1];
                --init_smpls (np.ndarray): (number of source, 85).
        """

        # 1. load the processed information by PreprocessConsumer
        src_infos = process_info.convert_to_src_info(self.opt.num_source)
        src_ids = src_infos["src_ids"][0]

        # 2. prepare the inputs information
        img_path = os.path.join(src_infos["img_dir"], src_infos["images"][src_ids])
        init_smpls = src_infos["smpls"][src_ids:src_ids+1]

        prepared_info = {
            "img_path": img_path,
            "init_smpls": init_smpls
        }
        return prepared_info


def human_estimate(opt) -> None:
    """

    Args:
        opt:

    Returns:

    """

    que = Queue()
    need_to_process = 0

    meta_src_proc = opt.meta_data["meta_src"]
    meta_ref_proc = opt.meta_data["meta_ref"]

    num_src = len(meta_src_proc)
    all_meta_proc = meta_src_proc + meta_ref_proc

    for i, meta_proc in enumerate(all_meta_proc):
        print(meta_proc)

        # check it is reference
        is_ref = (i >= num_src)

        if not meta_proc.check_has_been_processed(verbose=True):
            que.put((meta_proc, is_ref))
            need_to_process += 1

    if need_to_process > 0:
        MAX_PER_GPU_PROCESS = opt.Preprocess.MAX_PER_GPU_PROCESS
        per_gpu_process = int(np.ceil(need_to_process / len(opt.gpu_ids)))
        candidate_gpu_process = opt.gpu_ids * min(MAX_PER_GPU_PROCESS, per_gpu_process)
        num_gpu_process = min(len(candidate_gpu_process), need_to_process)

        consumers = []
        for gpu_process_id in range(num_gpu_process):
            gpu_id = candidate_gpu_process[gpu_process_id]
            consumer = PreprocessConsumer(
                que, gpu_id, opt
            )
            consumers.append(consumer)

        # all processors start
        for consumer in consumers:
            consumer.start()

        # all processors join
        for consumer in consumers:
            consumer.join()


def digital_deform(opt) -> None:
    """
        Digitalizing the source images.
    Args:
        opt:

    Returns:
        None
    """

    print("\t\tPre-processing: digital deformation start...")

    que = Queue()
    need_to_process = 0

    meta_src_proc = opt.meta_data["meta_src"]

    for i, meta_proc in enumerate(meta_src_proc):

        processed_info = ProcessInfo(meta_proc)
        processed_info.deserialize()

        if not HumanDigitalDeformConsumer.check_has_been_deformed(processed_info):
            que.put(processed_info)
            need_to_process += 1

    if need_to_process > 0:
        MAX_PER_GPU_PROCESS = opt.Preprocess.MAX_PER_GPU_PROCESS
        per_gpu_process = int(np.ceil(need_to_process / len(opt.gpu_ids)))
        candidate_gpu_process = opt.gpu_ids * min(MAX_PER_GPU_PROCESS, per_gpu_process)
        num_gpu_process = min(len(candidate_gpu_process), need_to_process)

        consumers = []
        for gpu_process_id in range(num_gpu_process):
            gpu_id = candidate_gpu_process[gpu_process_id]
            consumer = HumanDigitalDeformConsumer(
                que, gpu_id, opt
            )
            consumers.append(consumer)

        # all processors start
        for consumer in consumers:
            consumer.start()

        # all processors join
        for consumer in consumers:
            consumer.join()

    print("\t\tPre-processing: digital deformation completed...")


def post_update_opt(opt):
    """
    Post update the configurations based on the results of preprocessing.
    Args:
        opt:

    Returns:

    """

    meta_src_proc = opt.meta_data["meta_src"]
    valid_meta_src_proc = []

    cur_num_source = 1
    for meta_proc in meta_src_proc:
        process_info = ProcessInfo(meta_proc)
        process_info.deserialize()

        # check it has been processed successfully
        if process_info.check_has_been_processed(process_info.vid_infos, verbose=False):
            valid_meta_src_proc.append(meta_proc)
            num_source = process_info.num_sources()
            cur_num_source = max(cur_num_source, num_source)
        else:
            # otherwise, clean this inputs
            process_info.declare()

    meta_ref_proc = opt.meta_data["meta_ref"]
    valid_meta_ref_proc = []
    for meta_proc in meta_ref_proc:
        if meta_proc.check_has_been_processed(verbose=False):
            valid_meta_ref_proc.append(meta_proc)

    ## 3.1 update the personalization.txt
    checkpoints_dir = opt.meta_data["checkpoints_dir"]
    with open(os.path.join(checkpoints_dir, "personalization.txt"), "w") as writer:
        for meta_src in valid_meta_src_proc:
            writer.write(meta_src.primitives_dir + "\n")

    # update the number sources
    print(f"the current number of sources are {cur_num_source}, "
          f"while the pre-defined number of sources are {opt.num_source}. ")
    opt.num_source = min(cur_num_source, opt.num_source)

    # update the source information
    opt.meta_data["meta_src"] = valid_meta_src_proc

    # update the reference information
    opt.meta_data["meta_ref"] = valid_meta_ref_proc

    return opt


def preprocess(opt) -> bool:
    """
        Preprocess the source and target image path or video directory.
    Args:
        opt :

    Returns:
        successful (bool): Whether it processes successfully or not.
    """

    print("\tPre-processing: start...")

    # 1. human estimation, including 2D pose, tracking, 3D pose, parsing, and front estimation.
    human_estimate(opt=opt)

    # 2. digital deformation.
    digital_deform(opt=opt)

    # 3. post updating of options based on the pre-processed results.
    opt = post_update_opt(opt)

    successful = len(opt.meta_data["meta_src"]) > 0 and len(opt.meta_data["meta_ref"]) >= 0

    print(f"\tPre-processing: {'successfully' if successful else 'failed'}...")

    return successful


if __name__ == "__main__":
    from iPERCore.services.options.options_inference import InferenceOptions

    OPT = InferenceOptions().parse()

    # 1. pre-processing
    preprocess(opt=OPT)
