# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

from iPERCore.models import ModelsFactory
from iPERCore.tools.utils.multimedia.video import fuse_src_ref_multi_outputs
from iPERCore.services.preprocess import preprocess
from iPERCore.services.personalization import personalize
from iPERCore.services.options.process_info import ProcessInfo
from iPERCore.services.options.meta_info import MetaSwapImitateOutput
from iPERCore.services.base_runner import get_src_info_for_swapper_inference
from iPERCore.services.run_imitator import call_imitator_inference


def merge_all_source_processed_info(opt, meta_src_proc):
    # merge all source processed information
    vid_info_list = []
    for i, meta_src in enumerate(meta_src_proc):
        """
        meta_input:
                path: /p300/tpami/neuralAvatar/sources/fange_1/fange_1_ns=2
                bg_path: /p300/tpami/neuralAvatar/sources/fange_1/IMG_7225.JPG
                name: fange_1
        primitives_dir: ../tests/debug/primitives/fange_1
        processed_dir: ../tests/debug/primitives/fange_1/processed
        vid_info_path: ../tests/debug/primitives/fange_1/processed/vid_info.pkl
        """
        src_proc_info = ProcessInfo(meta_src)
        src_proc_info.deserialize()

        src_info = src_proc_info.convert_to_src_info(num_source=src_proc_info.num_sources())
        vid_info_list.append(src_info)

    src_info_for_inference = get_src_info_for_swapper_inference(opt, vid_info_list)

    # return src_info_for_inference
    return src_info_for_inference


def swap(opt):
    """

    Args:
        opt:

    Returns:
        all_meta_outputs (list of MetaOutput):

    """

    print("Step 3: running swapper.")

    if opt.ip:
        from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer
        visualizer = VisdomVisualizer(env=opt.model_id, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    # set imitator
    swapper = ModelsFactory.get_by_name("swapper", opt)

    # merge all sources
    meta_src_proc = opt.meta_data["meta_src"]
    src_info_for_inference = merge_all_source_processed_info(opt, meta_src_proc)

    # update number source
    opt.num_source = sum(src_info_for_inference["num_source"])
    print(f"update the number of sources {src_info_for_inference['num_source']} = {opt.num_source}")

    # source setup
    swapper.swap_source_setup(
        src_path_list=src_info_for_inference["paths"],
        src_smpl_list=src_info_for_inference["smpls"],
        masks_list=src_info_for_inference["masks"],
        bg_img_list=src_info_for_inference["bg"],
        offsets_list=src_info_for_inference["offsets"],
        links_ids_list=src_info_for_inference["links"],
        swap_parts=src_info_for_inference["swap_parts"],
        visualizer=visualizer,
        swap_masks=None
    )

    # call swap
    all_meta_outputs = []

    # check whether it has reference or not
    meta_ref_proc = opt.meta_data["meta_ref"]

    for j, meta_ref in enumerate(meta_ref_proc):
        """
        meta_input:
            path: /p300/tpami/neuralAvatar/references/videos/bantangzhuyi_1.mp4
            bg_path: 
            name: bantangzhuyi_1
            audio: /p300/tpami/neuralAvatar/references/videos/bantangzhuyi_1.mp3
            fps: 30.02
            pose_fc: 400.0
            cam_fc: 150.0
        primitives_dir: ../tests/debug/primitives/bantangzhuyi_1
        processed_dir: ../tests/debug/primitives/bantangzhuyi_1/processed
        vid_info_path: ../tests/debug/primitives/bantangzhuyi_1/processed/vid_info.pkl
        """
        meta_output = MetaSwapImitateOutput(meta_src_proc, meta_ref)

        ref_proc_info = ProcessInfo(meta_ref)
        ref_proc_info.deserialize()

        ref_info = ref_proc_info.convert_to_ref_info()

        results_dict = call_imitator_inference(
            opt, swapper, meta_output,
            ref_paths=ref_info["images"],
            ref_smpls=ref_info["smpls"],
            visualizer=visualizer
        )

        # save to video
        fuse_src_ref_multi_outputs(
            meta_output.out_mp4, src_info_for_inference["src_paths"],
            results_dict["ref_imgs_paths"], results_dict["outputs"],
            audio_path=meta_output.audio, fps=meta_output.fps,
            image_size=opt.image_size, pool_size=opt.num_workers
        )

        all_meta_outputs.append(meta_output)

    for meta_output in all_meta_outputs:
        print(meta_output)

    print("Step 3: running swapper done.")
    return all_meta_outputs


def run_swapper(opt):
    # 1. prepreocess
    successful = preprocess(opt)

    if successful:
        # 2. personalization
        personalize(opt)
        # 3. imitate
        all_meta_outputs = swap(opt)
    else:
        all_meta_outputs = []

    return all_meta_outputs


if __name__ == "__main__":
    from iPERCore.services.options.options_inference import InferenceOptions

    OPT = InferenceOptions().parse()
    run_swapper(opt=OPT)
