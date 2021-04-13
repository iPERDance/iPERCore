# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import argparse
from tqdm import tqdm
import torch
import torch.utils.data

from iPERCore.services.options.options_setup import setup
from iPERCore.data.dataset import DatasetFactory
from iPERCore.tools.trainers.base import FlowCompositionForTrainer
from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_ids", type=str, default="0", help="the gpu ids.")
parser.add_argument("--image_size", type=int, default=512, help="the image size.")
parser.add_argument("--model_id", type=str, default="visuals", help="use default, ignore this.")
parser.add_argument("--output_dir", type=str, default="./results", help="use default, ignore this.")
parser.add_argument("--dataset_dirs", type=str, nargs="*",
                    default=["/p300/tpami/datasets/fashionvideo", "/p300/tpami/datasets/iPER",
                             "/p300/tpami/datasets/motionSynthetic"])
parser.add_argument("--background_dir", type=str, default="/p300/tpami/places")
parser.add_argument("--dataset_mode", type=str, default="ProcessedVideo+Place2",
                    choices=["ProcessedVideo", "ProcessedVideo+Place2"])
parser.add_argument("--visdom_ip", type=str, default="http://10.10.10.100")
parser.add_argument("--visdom_port", type=int, default=31102)

args = parser.parse_args()
args.cfg_path = os.path.join("./assets", "configs", "trainers", "train_aug_bg.toml")
visualizer = VisdomVisualizer(env="visual", ip=args.visdom_ip, port=args.visdom_port)


def main():
    # set this as empty when preprocessing the training dataset.
    args.ref_path = ""

    cfg = setup(args)
    cfg.num_source = 4
    cfg.time_step = 2

    dataset = DatasetFactory.get_by_name(cfg.dataset_mode, cfg, is_for_train=True)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=4, pin_memory=True
    )

    device = torch.device("cuda:0")

    flow_comp = FlowCompositionForTrainer(cfg)
    flow_comp = flow_comp.to(device)

    for inputs in tqdm(dataloader):
        images = inputs["images"].to(device, non_blocking=True)
        smpls = inputs["smpls"].to(device, non_blocking=True)
        masks = inputs["masks"].to(device, non_blocking=True)
        offsets = inputs["offsets"].to(device, non_blocking=True) if "offsets" in inputs else 0
        links_ids = inputs["links_ids"].to(device, non_blocking=True) if "links_ids" in inputs else None
        aug_bg = inputs["bg"].to(device, non_blocking=True) if "bg" in inputs else None

        ns = cfg.num_source
        src_img = images[:, 0:ns].contiguous()
        src_smpl = smpls[:, 0:ns].contiguous()
        tsf_img = images[:, ns:].contiguous()
        tsf_smpl = smpls[:, ns:].contiguous()
        src_mask = masks[:, 0:ns].contiguous()
        ref_mask = masks[:, ns:].contiguous()

        ##################################################################
        # input_G_bg  (bs, 1, 4, 512, 512): for background inpainting network,
        # input_G_src (bs, ns, 6, 512, 512): for source identity network,
        # input_G_tsf (bs, nt, 6, 512, 512): for transfer network,
        # Tst (bs, ns, nt, 512, 512, 2):  the transformation flows from source (s_i) to target (t_j);
        # Ttt ():  if temporal is True, transformation from last time target (t_{j-1)
        #          to current time target (t_j), otherwise, it is None.
        #
        # src_mask  (bs, ns, 1, 512, 512): the source masks;
        # tsf_mask  (bs, nt, 1, 512, 512): the target masks;
        # head_bbox (ns, 4): the head bounding boxes of all targets;
        # body_bbox (ns, 4): the body bounding boxes of all targets;
        # uv_img (bs, 3, 512, 512): the extracted uv images, for visualization.
        ################################################################

        input_G_bg, input_G_src, input_G_tsf, Tst, Ttt, src_mask, tsf_mask, head_bbox, body_bbox, uv_img = \
            flow_comp(src_img, tsf_img, src_smpl, tsf_smpl, src_mask=src_mask, ref_mask=ref_mask,
                      links_ids=links_ids, offsets=offsets, temporal=cfg.temporal)

        visualizer.vis_named_img("input_G_bg", input_G_bg[0:1, 0, 0:3])
        visualizer.vis_named_img("input_G_src", input_G_src[0, :, 0:3])
        visualizer.vis_named_img("input_G_tsf", input_G_tsf[0, :, 0:3])
        visualizer.vis_named_img("src_mask", src_mask[0])
        visualizer.vis_named_img("tsf_mask", tsf_mask[0])
        visualizer.vis_named_img("src_img", src_img[0])
        visualizer.vis_named_img("tsf_img", tsf_img[0])
        visualizer.vis_named_img("uv_img", uv_img[0:1])

        if aug_bg is not None:
            visualizer.vis_named_img("aug_bg", aug_bg)
            print(f"aug_bg = {aug_bg.shape}")

        print(f"input_G_bg = {input_G_bg.shape}")
        print(f"input_G_src = {input_G_src.shape}")
        print(f"input_G_tsf = {input_G_tsf.shape}")
        print(f"Tst = {Tst.shape}")
        print(f"src_mask = {src_mask.shape}")
        print(f"tsf_mask = {tsf_mask.shape}")
        print(f"head_bbox = {head_bbox.shape}")
        print(f"body_bbox = {body_bbox.shape}")
        print(f"uv_img = {uv_img.shape}")


if __name__ == "__main__":
    main()
