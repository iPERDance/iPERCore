# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

from iPERCore.models import ModelsFactory
from iPERCore.tools.utils.signals.smooth import temporal_smooth_smpls
from iPERCore.tools.utils.filesio.persistence import clear_dir
from iPERCore.tools.utils.multimedia.video import fuse_src_ref_multi_outputs
from iPERCore.services.preprocess import preprocess
from iPERCore.services.personalization import personalize
from iPERCore.services.options.process_info import ProcessInfo
from iPERCore.services.options.meta_info import MetaImitateOutput
from iPERCore.services.base_runner import (
    get_src_info_for_inference,
    add_hands_params_to_smpl,
    add_special_effect,
    add_bullet_time_effect
)


def call_imitator_inference(opt, imitator, meta_output, ref_paths,
                            ref_smpls, visualizer, use_selected_f2pts=False):
    """

    Args:
        opt:
        imitator:
        meta_output:
        ref_paths:
        ref_smpls:
        visualizer:
        use_selected_f2pts:

    Returns:
        outputs (List[Tuple[str]]):
    """

    # if there are more than 10 frames, then we will use temporal smooth of smpl.
    if len(ref_smpls) > 10:
        ref_smpls = temporal_smooth_smpls(ref_smpls, pose_fc=meta_output.pose_fc, cam_fc=meta_output.cam_fc)

    out_imgs_dir = clear_dir(meta_output.out_img_dir)

    effect_info = meta_output.effect_info
    view_directions = effect_info["View"]
    bullet_time_list = effect_info["BT"]

    # check use multi-view outputs
    if len(view_directions) == 0:
        # if do not use multi-view outputs, only add bullet-time effects
        ref_smpls, ref_imgs_paths = add_bullet_time_effect(ref_smpls, ref_paths, bt_list=bullet_time_list)

        # add hands parameters to smpl
        ref_smpls = add_hands_params_to_smpl(ref_smpls, imitator.body_rec.np_hands_mean)

        # run imitator's inference function
        outputs = imitator.inference(tgt_smpls=ref_smpls, cam_strategy=opt.cam_strategy,
                                     output_dir=out_imgs_dir, prefix="pred_", visualizer=visualizer,
                                     verbose=True, use_selected_f2pts=use_selected_f2pts)
        outputs = list(zip(outputs))
    else:
        outputs = []
        ref_imgs_paths = ref_paths
        for i, view in enumerate(view_directions):
            # otherwise, we will add both multi-view and bullet-time effects
            ref_view_smpls, ref_imgs_paths = add_special_effect(ref_smpls, ref_paths,
                                                                view_dir=view, bt_list=bullet_time_list)

            # add hands parameters to smpl
            ref_view_smpls = add_hands_params_to_smpl(ref_view_smpls, imitator.body_rec.np_hands_mean)

            # run imitator's inference function
            view_outputs = imitator.inference(tgt_smpls=ref_view_smpls, cam_strategy=opt.cam_strategy,
                                              output_dir=out_imgs_dir, prefix=f"pred_{i}_{int(view)}_",
                                              visualizer=visualizer, verbose=True,
                                              use_selected_f2pts=use_selected_f2pts)
            outputs.append(view_outputs)

        outputs = list(zip(*outputs))

    results_dict = {
        "outputs": outputs,
        "ref_imgs_paths": ref_imgs_paths
    }

    return results_dict


def imitate(opt):
    """

    Args:
        opt:

    Returns:
        all_meta_outputs (list of MetaOutput):

    """

    print("Step 3: running imitator.")

    if opt.ip:
        from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer
        visualizer = VisdomVisualizer(env=opt.model_id, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    # set imitator
    imitator = ModelsFactory.get_by_name("imitator", opt)

    meta_src_proc = opt.meta_data["meta_src"]
    meta_ref_proc = opt.meta_data["meta_ref"]

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
        imitator.source_setup(
            src_path=src_info_for_inference["paths"],
            src_smpl=src_info_for_inference["smpls"],
            masks=src_info_for_inference["masks"],
            bg_img=src_info_for_inference["bg"],
            offsets=src_info_for_inference["offsets"],
            links_ids=src_info_for_inference["links"],
            visualizer=visualizer
        )

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
            meta_output = MetaImitateOutput(meta_src, meta_ref)

            ref_proc_info = ProcessInfo(meta_ref)
            ref_proc_info.deserialize()

            ref_info = ref_proc_info.convert_to_ref_info()

            results_dict = call_imitator_inference(
                opt, imitator, meta_output,
                ref_paths=ref_info["images"],
                ref_smpls=ref_info["smpls"],
                visualizer=visualizer
            )

            # save to video
            fuse_src_ref_multi_outputs(
                meta_output.out_mp4, src_info_for_inference["paths"],
                results_dict["ref_imgs_paths"], results_dict["outputs"],
                audio_path=meta_output.audio, fps=meta_output.fps,
                image_size=opt.image_size, pool_size=opt.num_workers
            )

            all_meta_outputs.append(meta_output)

    for meta_output in all_meta_outputs:
        print(meta_output)

    print("Step 3: running imitator done.")
    return all_meta_outputs


def run_imitator(opt):
    # 1. prepreocess
    successful = preprocess(opt)

    if successful:
        # 2. personalization
        personalize(opt)
        # 3. imitate
        all_meta_outputs = imitate(opt)
    else:
        all_meta_outputs = []

    return all_meta_outputs


if __name__ == "__main__":
    from iPERCore.services.options.options_inference import InferenceOptions

    OPT = InferenceOptions().parse()
    run_imitator(opt=OPT)
