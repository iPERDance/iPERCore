# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

from iPERCore.models import ModelsFactory
from iPERCore.tools.utils.filesio.persistence import clear_dir
from iPERCore.tools.utils.multimedia.video import fuse_source_output
from iPERCore.services.preprocess import preprocess
from iPERCore.services.personalization import personalize
from iPERCore.services.options.process_info import ProcessInfo
from iPERCore.services.options.meta_info import MetaNovelViewOutput
from iPERCore.services.base_runner import (
    add_hands_params_to_smpl,
    create_T_pose_novel_view_smpl,
    get_src_info_for_inference
)


def novel_view(opt):
    """

    Args:
        opt:

    Returns:
        all_meta_outputs (list of MetaOutput):

    """

    print("Step 3: running novel viewer.")

    if opt.ip:
        from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer
        visualizer = VisdomVisualizer(env=opt.model_id, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    # set imitator
    viewer = ModelsFactory.get_by_name("viewer", opt)

    meta_src_proc = opt.meta_data["meta_src"]

    all_meta_outputs = []
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

        src_info = src_proc_info.convert_to_src_info(num_source=opt.num_source)
        src_info_for_inference = get_src_info_for_inference(opt, src_info)

        # source setup
        viewer.source_setup(
            src_path=src_info_for_inference["paths"],
            src_smpl=src_info_for_inference["smpls"],
            masks=src_info_for_inference["masks"],
            bg_img=src_info_for_inference["bg"],
            offsets=src_info_for_inference["offsets"],
            links_ids=src_info_for_inference["links"],
            visualizer=visualizer
        )

        novel_smpls = create_T_pose_novel_view_smpl(length=180)
        novel_smpls[:, -10:] = src_info_for_inference["smpls"][0, -10:]

        if not opt.T_pose:
            novel_smpls[:, 6:-10] = src_info_for_inference["smpls"][0, 6:-10]

        novel_smpls = add_hands_params_to_smpl(novel_smpls, viewer.body_rec.np_hands_mean)
        meta_output = MetaNovelViewOutput(meta_src)

        out_imgs_dir = clear_dir(meta_output.out_img_dir)
        outputs = viewer.inference(tgt_smpls=novel_smpls, cam_strategy="smooth",
                                   output_dir=out_imgs_dir, visualizer=visualizer, verbose=True)

        fuse_source_output(
            meta_output.out_mp4, src_info_for_inference["paths"],
            outputs, audio_path=None, fps=25, image_size=opt.image_size, pool_size=opt.num_workers
        )

        all_meta_outputs.append(meta_output)

    for meta_output in all_meta_outputs:
        print(meta_output)

    print("Step 3: running novel viewer done.")
    return all_meta_outputs


def run_viewer(opt):
    # 1. prepreocess
    successful = preprocess(opt)

    if successful:
        # 2. personalization
        personalize(opt)
        # 3. imitate
        all_meta_outputs = novel_view(opt)
    else:
        all_meta_outputs = []

    return all_meta_outputs


if __name__ == "__main__":
    from iPERCore.services.options.options_inference import InferenceOptions

    OPT = InferenceOptions().parse()
    run_viewer(opt=OPT)
