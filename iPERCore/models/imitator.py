# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .networks import NetworksFactory
from .base_model import BaseRunnerModel
from .flowcomposition import FlowComposition, FlowCompositionForSwapper

from iPERCore.tools.human_digitalizer.bodynets import SMPL, SMPLH
from iPERCore.tools.utils.filesio import cv_utils
from iPERCore.tools.utils.geometry import cam_pose_utils


class TemporalFIFO(object):
    def __init__(self, time_step, n_down=3, n_res_block=6):
        self.time_step = time_step
        self.n_down = n_down

        temporal_info = dict()
        temporal_info["fim"] = [0] * self.time_step
        temporal_info["wim"] = [0] * self.time_step
        temporal_info["f2pts"] = [0] * self.time_step

        self.temporal_info = temporal_info
        self.temporal_enc = [[0] * self.time_step for i in range(n_down)]
        self.temporal_res = [[0] * self.time_step for i in range(n_res_block)]

        self.temporal_preds = [0] * self.time_step

        self.index = 0

    def append_info(self, smpl_info=None, preds=None, tsf_enc_outs=None, tsf_res_outs=None):
        """
            1. add the smpl_info to self.temporal_info, the details are
                --fim:     [(bs, h, w), (bs, h, w), ..., (bs, h, w)], the length is self.time_step
                --wim:     [(bs, h, w, 3), (bs, h, w, 3), ..., (bs, h, w, 3)], the length is self.time_step
                --f2pts: [(bs, 13776, 3, 3), (bs, 13776, 3, 3), ..., (bs, 13776, 3, 3)], the length is self.time_step

            2. add the features of encoders to self.temporal_enc, the details are
                self.temporal_enc[0] = [(bs, c1, h1, w1), ..., (bs, c1, h1, w1)],the length is self.time_step
                self.temporal_enc[1] = [(bs, c2, h2, w2), ..., (bs, c2, h2, w2)],the length is self.time_step
                ...
                self.temporal_enc[n_down of encoder] = [(bs, cn, hn, wn), ...],the length is self.time_step
        Args:
            smpl_info (dict): the dict of smpl information.
                --fim (torch.tensor):     (bs, h, w)
                --wim (torch.tensor):     (bs, h, w, 3)
                --f2pts (torch.tensor):   (bs, 13776, 3, 3)

            tsf_enc_outs (list of torch.tensor):
        Returns:
            None
        """

        fim = smpl_info["fim"]
        wim = smpl_info["wim"]
        f2pts = smpl_info["f2pts"]

        i = self.index % self.time_step

        self.temporal_info["fim"][i] = fim
        self.temporal_info["wim"][i] = wim
        self.temporal_info["f2pts"][i] = f2pts

        if tsf_enc_outs is not None:
            for s_i, tsf_enc in enumerate(tsf_enc_outs):
                self.temporal_enc[s_i][i] = tsf_enc

        if tsf_res_outs is not None:
            for s_i, tsf_res in enumerate(tsf_res_outs):
                self.temporal_res[s_i][i] = tsf_res

        self.temporal_preds[i] = preds
        self.index += 1

    def temporal_enc_outs_to_tensor(self):
        fused_enc_outs = []
        for temp_enc_x in self.temporal_enc:
            # temp_enc_x (list of torch.tensor): [(bs, c1, h1, w1), (bs, c1, h1, w1), ..., (bs, c1, h1, w1)]
            if self.is_full:
                # print(self.index, len(temp_enc_x[0:self.index]))
                fused_enc_outs.append(torch.cat(temp_enc_x, dim=0))  # (bs * nt, c1, h1, w1)
            else:
                # print(self.index, len(temp_enc_x[0:self.index]))
                fused_enc_outs.append(torch.cat(temp_enc_x[0:self.index], dim=0))  # (bs * nt, c1, h1, w1)

        fused_res_outs = []
        for temp_res_x in self.temporal_res:
            # temp_enc_x (list of torch.tensor): [(bs, c1, h1, w1), (bs, c1, h1, w1), ..., (bs, c1, h1, w1)]

            if self.is_full:
                fused_res_outs.append(torch.cat(temp_res_x, dim=0))  # (bs * nt, c1, h1, w1)
            else:
                fused_res_outs.append(torch.cat(temp_res_x[0:self.index], dim=0))  # (bs * nt, c1, h1, w1)

        return fused_enc_outs, fused_res_outs

    def temporal_info_to_tensor(self):
        smpl_info = dict()
        for key, values in self.temporal_info.items():
            # print(key, len(values))
            if self.is_full:
                smpl_info[key] = torch.cat(values, dim=0)
            else:
                smpl_info[key] = torch.cat(values[0:self.index], dim=0)

        return smpl_info

    def temporal_preds_tensor(self):
        if self.is_full:
            preds = torch.cat(self.temporal_preds, dim=0)
        else:
            preds = torch.cat(self.temporal_preds[0:self.index], dim=0)

        return preds

    @property
    def nt(self):
        return min(self.index, self.time_step)

    @property
    def is_full(self):
        return self.nt == self.time_step


class Imitator(BaseRunnerModel):
    def __init__(self, opt, device=torch.device("cuda:0")):
        super(Imitator, self).__init__(opt)
        self._name = "Imitator"
        self.device = device

        self.src_info = None
        self.tsf_info = None
        self.first_cam = None

        self._create_networks()

    def _create_networks(self):
        # 1. body mesh recovery model
        # self.body_rec = SMPL(self._opt.smpl_model).to(self.device)
        self.body_rec = SMPLH(model_path=self._opt.smpl_model_hand).to(self.device)

        self.weak_cam_swapper = cam_pose_utils.WeakPerspectiveCamera(self.body_rec)

        # 2. flow composition module
        self.flow_comp = FlowComposition(opt=self._opt).to(self.device)

        # 3.0 create generator
        self.generator, self.temporal_fifo = self._create_generator(
            self._opt.neural_render_cfg.Generator)

        self.generator = self.generator.to(self.device)

    def _create_generator(self, cfg):
        gen_name = self._opt.gen_name
        net = NetworksFactory.get_by_name(gen_name, cfg=cfg, temporal=self._opt.temporal)

        if os.path.exists(self._opt.meta_data.personalized_ckpt_path):
            load_path = self._opt.meta_data.personalized_ckpt_path
        else:
            load_path = self._opt.load_path_G

        ckpt = torch.load(load_path, map_location="cpu")
        net.load_state_dict(ckpt, strict=False)
        net.eval()

        print(f"Loading net from {load_path}")

        temporal_fifo = TemporalFIFO(self._opt.time_step, len(cfg.TSFNet.num_filters), cfg.TSFNet.n_res_block)

        return net, temporal_fifo

    @torch.no_grad()
    def source_setup(self, src_path, src_smpl, masks=None, bg_img=None, offsets=0, links_ids=None, visualizer=None):
        """
            pre-process the source information
        Args:
            src_path (list of str): the source image paths, len(src_path) = ns
            src_smpl (torch.tensor or np.ndarray)): (ns, 85)
            masks (list of np.ndarray): [(1, h, w), (1, h, w), ..., (1, h, w)] or (ns, 1, h, w)
            bg_img (torch.tensor): (3, h, w)
            offsets (np.ndarray or 0): (ns, nv, 3) or (nv, 3)
            links_ids (np.ndarray or None): (nv,)
            visualizer (Visualizer or None):

        Returns:
            src_info (dict): the source information.

        """
        # 1. load source images (1, ns, 3, H, W)
        src_img = torch.tensor(cv_utils.load_images(src_path, self._opt.image_size)[None]).float().to(self.device)

        # 2. process source inputs for (bg_net and src_net)
        if isinstance(src_smpl, np.ndarray):
            src_smpl = torch.tensor(src_smpl).float().to(self.device)
        elif src_smpl.device != self.device:
            src_smpl = src_smpl.to(self.device)

        # 2.1 the source smpl information
        offsets = torch.tensor(offsets).float().to(self.device)
        src_info = self.body_rec.get_details(src_smpl, offsets, links_ids=links_ids)
        src_info["num_source"] = src_smpl.shape[0]

        if masks is not None:
            src_info["masks"] = 1.0 - torch.tensor(masks).float().to(self.device)
        self.flow_comp.add_rendered_f2verts_fim_wim(src_info, use_morph=True, get_uv_info=True)

        src_info["offsets"] = offsets
        src_info["links_ids"] = links_ids

        # 2.2 flow composition
        uv_img, input_G_bg, input_G_src = self.flow_comp.process_source(src_img, src_info, primary_ids=[0])
        src_info["uv_img"] = uv_img

        # 3. background inpaintor
        if self._opt.use_inpaintor or bg_img is not None:
            bg_img = torch.tensor(bg_img).float()
            bg_img.unsqueeze_(0)
            bg_img.unsqueeze_(0)
            bg_img = bg_img.to(self.device)
        else:
            bg_img = self.generator.forward_bg(input_G_bg)  # (N, ns, 3, h, w)
            # replace visible parts
            # bg_img = input_G_bg[:, :, 0:3] + (1 - input_G_bg[:, :, -1:]) * bg_img

        # 3. process source inputs
        # src_enc_outs: [torch.tensor(bs*ns, c1, h1, w1), tensor.tensor(bs*ns, c2, h2, w2), ... ]
        src_enc_outs, src_res_outs = self.generator.forward_src(input_G_src, only_enc=True)

        src_info["img"] = src_img
        src_info["bg"] = bg_img[:, 0].contiguous()
        src_info["feats"] = (src_enc_outs, src_res_outs)

        self.src_info = src_info

        if visualizer is not None:
            visualizer.vis_named_img("bg", src_info["bg"])
            visualizer.vis_named_img("src_img", src_info["img"][0])
            visualizer.vis_named_img("src_cond", src_info["cond"])
            visualizer.vis_named_img("uv_img", src_info["uv_img"])

        return src_info

    def swap_params(self, src_cam, src_shape, tgt_smpl, cam_strategy="smooth"):
        tgt_cam = tgt_smpl[:, 0:3]
        pose = tgt_smpl[:, 3:-10]

        cam = self.weak_cam_swapper.cam_swap(src_cam, tgt_cam, self.first_cam, cam_strategy)

        ref_smpl = torch.cat([cam, pose, src_shape], dim=1)

        return ref_smpl

    @torch.no_grad()
    def make_inputs_for_tsf(self, src_info, tgt_smpl, cam_strategy="smooth", t=0,
                            primary_ids=0, use_selected_f2pts=False):
        """
            process the inputs for tsf_net
        Args:
            src_info     (dict): the source setup information, it contains the followings:
                --cam       (torch.Tensor):         (ns, 3);
                --shape     (torch.Tensor):         (ns, 10);
                --pose      (torch.Tensor):         (ns, 72);
                --fim       (torch.Tensor):         (1 * ns, h, w),
                --wim       (torch.Tensor):         (1 * ns, h, w, 3),
                --f2pts     (torch.Tensor):         (1 * ns, 13776, 3, 2)
                --selected_f2pts (torch.Tensor):    (1 * ns, 13776, 3, 2)
                --only_vis_f2pts (torch.Tensor):    (1 * ns, 13776, 3, 2)
                --feats     (Tuple[List]): ([(ns, c1, h1, w2), ..., (ns, ck, hk, wk)],
                                            [(ns, ck, hk, wk), ..., (ns, ck, hk, wk)])

                --offsets   (torch.Tensor or 0):    (num_verts, 3) or 0;
                --links_ids (torch.Tensor or None): (num_verts, 3) or None;
                --uv_img    (torch.Tensor):         (1, 3, h, w);
                --bg        (torch.Tensor):         (1, 3, h, w);

            tgt_smpl     (torch.Tensor) :  (nt, 85)
            cam_strategy (str): "smooth"
            t            (int):
            primary_ids  (int):
            use_selected_f2pts (bool):

        Returns:
            input_G_tsf   (torch.Tensor):
            Tst           (torch.Tensor):
            Ttt           (torch.Tensor or None):
            temp_enc_outs (list of torch.Tensor):
            ref_info      (dict):
        """
        bs = 1
        ns = src_info["num_source"]

        # 1. swap the params of smpl
        if t == 0 and cam_strategy == "smooth":
            self.first_cam = tgt_smpl[:, 0:3].clone()

        ref_smpl = self.swap_params(
            src_info["cam"][primary_ids:primary_ids + 1],
            src_info["shape"][primary_ids:primary_ids + 1],
            tgt_smpl, cam_strategy
        )

        ref_info = self.body_rec.get_details(ref_smpl, src_info["offsets"], links_ids=src_info["links_ids"])
        self.flow_comp.add_rendered_f2verts_fim_wim(ref_info, use_morph=False, get_uv_info=False)

        # 2. make inputs for tsf
        input_G_tsf = self.flow_comp.make_tsf_inputs(src_info["uv_img"], ref_info)

        # 3. calculate the transformation flow
        if t == 0 or not self._opt.temporal:
            Tst, Ttt = self.flow_comp.make_trans_flow(bs, ns, 1, src_info, None, ref_info,
                                                      temporal=False, use_selected_f2pts=use_selected_f2pts)
            temp_enc_outs, temp_res_outs = None, None
        else:
            nt = self.temporal_fifo.nt
            temp_info = self.temporal_fifo.temporal_info_to_tensor()
            temp_enc_outs, temp_res_outs = self.temporal_fifo.temporal_enc_outs_to_tensor()
            Tst, Ttt = self.flow_comp.make_trans_flow(bs, ns, nt, src_info, temp_info,
                                                      ref_info, temporal=True, use_selected_f2pts=use_selected_f2pts)

        return input_G_tsf, Tst, Ttt, temp_enc_outs, temp_res_outs, ref_info

    @torch.no_grad()
    def inference(self, tgt_smpls, cam_strategy="smooth", output_dir="", prefix="pred_",
                  use_selected_f2pts=False, visualizer=None, verbose=True):

        outputs = []
        length = len(tgt_smpls)
        process_bar = tqdm(range(length), desc=prefix) if verbose else range(length)

        self.first_cam = None

        tgt_smpls = torch.tensor(tgt_smpls).float().to(self.device)
        if cam_strategy == "smooth":
            tgt_smpls = self.weak_cam_swapper.stabilize(tgt_smpls)

        for t in process_bar:
            tgt_smpl = tgt_smpls[t:t+1]
            input_G_tsf, Tst, Ttt, temp_enc_outs, temp_res_outs, ref_info = self.make_inputs_for_tsf(
                self.src_info, tgt_smpl, cam_strategy, t, use_selected_f2pts=use_selected_f2pts
            )

            preds, tsf_mask = self.forward(input_G_tsf[:, 0], Tst, temp_enc_outs, temp_res_outs, Ttt)

            if t != 0 and self._opt.temporal:
                prev_preds = self.temporal_fifo.temporal_preds_tensor()
                preds_warp = F.grid_sample(prev_preds, Ttt.view(-1, self._opt.image_size, self._opt.image_size, 2))
            else:
                preds_warp = None

            if visualizer is not None:
                src_warp = F.grid_sample(self.src_info["img"].view(-1, 3, self._opt.image_size, self._opt.image_size),
                                         Tst.view(-1, self._opt.image_size, self._opt.image_size, 2))
                visualizer.vis_named_img("pred_" + cam_strategy, preds)
                visualizer.vis_named_img("uv_warp", input_G_tsf[0, :, 0:3])
                visualizer.vis_named_img("src_warp", src_warp)

                if preds_warp is not None:
                    visualizer.vis_named_img("preds_warp", preds_warp)

            if self._opt.temporal:
                self.post_update(ref_info, preds)

            if output_dir:
                filename = "{:0>8}.png".format(t)
                preds = preds[0].cpu().numpy()
                file_path = os.path.join(output_dir, prefix + filename)
                cv_utils.save_cv2_img(preds, file_path, normalize=True)

                outputs.append(file_path)
            else:
                preds = preds[0].cpu().numpy()
                outputs.append(preds)
                # tsf_mask = tsf_mask[0, 0].cpu().numpy() * 255
                # tsf_mask = tsf_mask.astype(np.uint8)
                # cv_utils.save_cv2_img(tsf_mask, os.path.join(output_dir, "mask_" + filename), normalize=False)

        return outputs

    def forward(self, tsf_inputs, Tst, temp_enc_outs=None, temp_res_outs=None, Ttt=None):
        bg_img = self.src_info["bg"]  # (bs, 3, h, w)
        src_enc_outs, src_res_outs = self.src_info["feats"]  # [(bs * ns, c1, h1, w1), ..., (bs * ns, c2, h2, w2)]

        tsf_img, tsf_mask = self.generator.forward_tsf(
            tsf_inputs, src_enc_outs, src_res_outs, Tst,
            temp_enc_outs=temp_enc_outs, temp_res_outs=temp_res_outs, Ttt=Ttt
        )

        pred_imgs = tsf_mask * bg_img + (1 - tsf_mask) * tsf_img

        return pred_imgs, tsf_mask

    def post_update(self, ref_info, preds):
        cur_inputs = torch.cat([preds, ref_info["cond"]], dim=1).unsqueeze_(dim=1)
        tsf_enc_outs, tsf_res_outs = self.generator.forward_src(cur_inputs, only_enc=True)

        self.temporal_fifo.append_info(ref_info, preds, tsf_enc_outs, tsf_res_outs)


class Viewer(Imitator):
    def __init__(self, opt, device=torch.device("cuda:0")):
        super(Viewer, self).__init__(opt, device)
        self._name = "Viewer"

    @torch.no_grad()
    def inference(self, tgt_smpls, cam_strategy="smooth",
                  output_dir="", visualizer=None, verbose=True):

        outputs = []

        length = len(tgt_smpls)
        process_bar = tqdm(range(length)) if verbose else range(length)

        self.first_cam = None

        tgt_smpls = torch.tensor(tgt_smpls).float().to(self.device)
        if cam_strategy == "smooth":
            tgt_smpls = self.weak_cam_swapper.stabilize(tgt_smpls)

        for t in process_bar:
            tgt_smpl = tgt_smpls[t:t+1]
            input_G_tsf, Tst, Ttt, temp_enc_outs, temp_res_outs, ref_info = self.make_inputs_for_tsf(
                self.src_info, tgt_smpl, cam_strategy, t, use_selected_f2pts=False
            )

            preds, tsf_mask = self.forward(input_G_tsf[:, 0], Tst, temp_enc_outs, temp_res_outs, Ttt)

            if t != 0 and self._opt.temporal:
                prev_preds = self.temporal_fifo.temporal_preds_tensor()
                preds_warp = F.grid_sample(prev_preds, Ttt.view(-1, self._opt.image_size, self._opt.image_size, 2))
            else:
                preds_warp = None

            if visualizer is not None:
                src_warp = F.grid_sample(self.src_info["img"].view(-1, 3, self._opt.image_size, self._opt.image_size),
                                         Tst.view(-1, self._opt.image_size, self._opt.image_size, 2))
                visualizer.vis_named_img("pred_" + cam_strategy, preds)
                visualizer.vis_named_img("uv_warp", input_G_tsf[0, :, 0:3])
                visualizer.vis_named_img("src_warp", src_warp)

                if preds_warp is not None:
                    visualizer.vis_named_img("preds_warp", preds_warp)

            if self._opt.temporal:
                self.post_update(ref_info, preds)

            if output_dir:
                filename = "{:0>8}.png".format(t)
                preds = preds[0].cpu().numpy()
                file_path = os.path.join(output_dir, "pred_" + filename)
                cv_utils.save_cv2_img(preds, file_path, normalize=True)

                outputs.append(file_path)
            else:
                preds = preds[0].cpu().numpy()
                outputs.append(preds)
                # tsf_mask = tsf_mask[0, 0].cpu().numpy() * 255
                # tsf_mask = tsf_mask.astype(np.uint8)
                # cv_utils.save_cv2_img(tsf_mask, os.path.join(output_dir, "mask_" + filename), normalize=False)

        return outputs


class Swapper(Imitator):
    def __init__(self, opt, device=torch.device("cuda:0")):
        super(Swapper, self).__init__(opt, device)
        self._name = "Swapper"

    def _create_networks(self):
        # 1. body mesh recovery model
        # self.body_rec = SMPL(self._opt.smpl_model).to(self.device)
        self.body_rec = SMPLH(model_path=self._opt.smpl_model_hand).to(self.device)

        self.weak_cam_swapper = cam_pose_utils.WeakPerspectiveCamera(self.body_rec)

        # 2. flow composition module
        self.flow_comp = FlowCompositionForSwapper(opt=self._opt).to(self.device)

        # 3.0 create generator
        self.generator, self.temporal_fifo = self._create_generator(
            self._opt.neural_render_cfg.Generator)

        self.generator = self.generator.to(self.device)

    def get_selected_info_by_part_mask(self, swap_masks):
        """
        Get the selected face index by annotated part masks.

        Args:
            swap_masks (List[List[np.ndarray]]):

        Returns:
            --selected_part_ids (List[List[int]]):
            --selected_face_ids (List[List[int]]):
        """
        raise NotImplementedError

    def get_selected_info_by_part_name(self, swap_parts, primary_ids=0):
        """
            Each source image might provides multiple parts, and here we need to calculate the parts of each source.
        Args:
            swap_parts (List[List[str]]): the part names of each source.
            primary_ids (int):

        Returns:
            --selected_part_ids (List[List[int]]):
            --selected_face_ids (List[List[int]]):
        """

        selected_part_ids = []
        selected_face_ids = []
        selected_face_ids_set = set()
        selected_part_ids_set = set()
        for swap_part in swap_parts:

            each_swap_part_ids = set()
            each_swap_face_ids = set()
            for sub_part in swap_part:
                part_ids = self.flow_comp.PART_IDS[sub_part]
                face_ids = self.flow_comp.get_selected_fids(part_ids)

                each_swap_part_ids |= set(part_ids)
                each_swap_face_ids |= set(face_ids)

            selected_face_ids_set |= each_swap_face_ids
            selected_part_ids_set |= each_swap_part_ids

            selected_part_ids.append(list(each_swap_part_ids))
            selected_face_ids.append(list(each_swap_face_ids))

        left_face_ids = set(self.flow_comp.all_faces_ids) - selected_face_ids_set

        if len(left_face_ids) > 0:
            primary_face_ids = set(selected_face_ids[primary_ids])
            primary_face_ids |= left_face_ids

            print(f"{len(left_face_ids)} faces have not been selected, "
                  f"and we will take all of them into primary faces. ")

            selected_face_ids[primary_ids] = list(primary_face_ids)

        return selected_part_ids, selected_face_ids

    def swap_source_setup(self, src_path_list, src_smpl_list, masks_list, bg_img_list=None,
                          offsets_list=0, links_ids_list=None, swap_parts=(["head"], ["body"]),
                          swap_masks=None, primary_ids=0, visualizer=None):
        """
            pre-process the source information
        Args:
            src_path_list  (List[List[str]]): the source image paths, len(src_path_list) = the number of people;
            src_smpl_list  (List[Union[torch.tensor,np.ndarray]]): [(ns_1, 85), ..., (ns_p, 85)];
            masks_list     (List[Union[np.ndarray, None]]): [(ns_1, 1, h, w), ..., None, ..., (ns_p, 1, h, w)]
            bg_img_list    (List[Union[torch.Tensor, None]]): (3, h, w)
            offsets_list   (List[np.ndarray]): [(num_verts, 3), ..., (num_verts, 3)] or 0;
            links_ids_list (List[Union[np.ndarray, None]]): [(num_verts, 3), ..., None, ..., (num_verts, 3)];
            swap_parts     (List[List[str]]):
            swap_masks     (Union[None, List[np.ndarray]]):
            primary_ids   (int):
            visualizer     (Visualizer or None):

        Returns:
            src_info (dict): the source information.
        """

        assert not (swap_parts is None and swap_masks is None)

        if swap_parts is not None:
            selected_part_ids, selected_face_ids = self.get_selected_info_by_part_name(swap_parts)
        else:
            selected_part_ids, selected_face_ids = self.get_selected_info_by_part_mask(swap_masks)

        src_info_list = []
        num_people = len(src_path_list)

        for i in range(num_people):
            src_path = src_path_list[i]
            src_smpl = src_smpl_list[i]
            masks = masks_list[i]
            bg_img = bg_img_list[i]
            offsets = offsets_list[i]
            links_ids = links_ids_list[i]

            src_info = self.source_setup(src_path, src_smpl, masks, bg_img, offsets=offsets,
                                         links_ids=links_ids, visualizer=visualizer)

            face_ids = []
            for _ in range(src_info["num_source"]):
                face_ids.append(selected_face_ids[i])

            self.flow_comp.add_rendered_selected_f2pts(src_info, face_ids)
            src_info_list.append(src_info)

            # if visualizer is not None:
            #     visualizer.vis_named_img(f"uv_img_{i}", src_info["uv_img"])

        merge_src_info = self.flow_comp.merge_src_info(src_info_list, primary_ids=primary_ids)
        self.src_info = merge_src_info

        if visualizer is not None:
            visualizer.vis_named_img("uv_img", merge_src_info["uv_img"])

        # print(merge_src_info["img"].shape)
        # print(merge_src_info["cam"].shape)
        # print(merge_src_info["shape"].shape)
        # print(merge_src_info["pose"].shape)
        # print(merge_src_info["fim"].shape)
        # print(merge_src_info["wim"].shape)
        # print(merge_src_info["f2pts"].shape)
        # print(merge_src_info["selected_f2pts"].shape)
        # print(merge_src_info["only_vis_f2pts"].shape)
        #
        # for merge_feats in merge_src_info["feats"][0]:
        #     print(merge_feats.shape)
        #
        # for merge_feats in merge_src_info["feats"][1]:
        #     print(merge_feats.shape)

        return merge_src_info
