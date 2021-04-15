# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
from torch.nn import functional as F
import numpy as np

from iPERCore.tools.utils.morphology import CannyFilter
from iPERCore.tools.utils.morphology import morph
from iPERCore.tools.human_digitalizer.renders import SMPLRenderer
from iPERCore.tools.utils.geometry import mesh


# from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer
#
# visualizer = VisdomVisualizer(
#     env="morph",
#     ip="http://10.10.10.100", port=31102
# )


class FlowComposition(torch.nn.Module):

    PART_IDS = {
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
        "upper": [1, 4, 5, 8, 9],
        "lower": [2, 3, 6, 7],
        "body": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    def __init__(self, opt):
        super(FlowComposition, self).__init__()
        self._opt = opt
        self._name = "FlowComposition"

        self._init_create_networks()

        for param in self.parameters():
            param.requires_grad = False

        self.f2uvs = None
        self.uv_fim = None
        self.uv_wim = None
        self.one_map = None

        # for Swapper
        self.part_faces = None
        self.uv_parts = None

    def _create_render(self):
        render = SMPLRenderer(
            face_path=self._opt.face_path,
            fim_enc_path=self._opt.fim_enc_path,
            uv_map_path=self._opt.uv_map_path,
            part_path=self._opt.part_path,
            map_name=self._opt.map_name,
            image_size=self._opt.image_size, fill_back=False, anti_aliasing=True,
            background_color=(0, 0, 0), has_front=True, top_k=3
        )

        return render

    def _init_create_networks(self):
        # hmr and render
        self.render = self._create_render()
        self.register_buffer("grid", self.render.create_meshgrid(image_size=self._opt.image_size))

    def make_uv_setup(self, bs, ns, nt, device):
        if self.f2uvs is None:
            uv_fim, uv_wim = self.render.render_uv_fim_wim(bs * max(ns, nt))
            self.f2uvs = self.render.get_f_uvs2img(bs * max(ns, nt))
            self.uv_fim = uv_fim
            self.uv_wim = uv_wim
            self.one_map = torch.ones(bs * ns, 1, self._opt.image_size, self._opt.image_size,
                                      dtype=torch.float32).to(device)

    def make_uv_img(self, src_img, src_info):
        """
        Args:
            src_img (torch.tensor): (bs, ns, 3, h, w)
            src_info (dict):

        Returns:
            merge_uv (torch.tensor): (bs, 3, h, w)
        """

        bs, ns, _, h, w = src_img.shape
        bsxns = bs * ns

        ## previous
        # only_vis_src_f2pts = src_info["only_vis_f2pts"]
        # Ts2uv = self.render.cal_bc_transform(only_vis_src_f2pts, self.uv_fim[0:bsxns], self.uv_wim[0:bsxns])
        # src_warp_to_uv = F.grid_sample(src_img.view(bs * ns, 3, h, w), Ts2uv)
        # vis_warp_to_uv = F.grid_sample(self.one_map, Ts2uv)
        # merge_uv = torch.sum(src_warp_to_uv.view(bs, ns, -1, h, w), dim=1) / (
        #     torch.sum(vis_warp_to_uv.view(bs, ns, -1, h, w), dim=1) + 1e-5)

        ## current
        uv_fim = self.uv_fim[0:bsxns]
        uv_wim = self.uv_wim[0:bsxns]
        one_map = self.one_map[0:bsxns]
        only_vis_src_f2pts = src_info["only_vis_obj_f2pts"]
        src_f2pts = src_info["obj_f2pts"]
        only_vis_Ts2uv = self.render.cal_bc_transform(only_vis_src_f2pts, uv_fim, uv_wim)
        Ts2uv = self.render.cal_bc_transform(src_f2pts, uv_fim, uv_wim)

        src_warp_to_uv = F.grid_sample(src_img.view(bs * ns, 3, h, w), Ts2uv).view(bs, ns, -1, h, w)
        vis_warp_to_uv = F.grid_sample(one_map, only_vis_Ts2uv)

        # TODO, here ks=13 is hyper-parameter, might need to set it to the configuration.
        vis_warp_to_uv = morph(vis_warp_to_uv, ks=13, mode="dilate").view(bs, ns, -1, h, w)

        vis_sum = torch.sum(vis_warp_to_uv[:, 1:], dim=1)
        temp = torch.sum(src_warp_to_uv[:, 1:] * vis_warp_to_uv[:, 1:], dim=1) / (vis_sum + 1e-5)

        vis_front = vis_warp_to_uv[:, 0]
        vis_other = (vis_sum >= 1).float()

        front_invisible = (1 - vis_front) * vis_other
        merge_uv = src_warp_to_uv[:, 0] * (1 - front_invisible) + temp * front_invisible

        # merge_uv = src_warp_to_uv[:, 0]
        # noisy = torch.randn((bs, 3, h, w), dtype=torch.float32).to(src_img.device)
        # merge_uv = 0.5 * merge_uv + 0.5 * noisy
        # merge_uv = torch.clamp(merge_uv, min=-1.0, max=1.0)

        return merge_uv

    def add_rendered_f2verts_fim_wim(self, smpl_info, use_morph=False, get_uv_info=True):
        """
        Args:
            smpl_info (dict): the smpl information contains:
                --cam (torch.Tensor):
                --verts (torch.Tensor):

            use_morph (bool): use morphing strategy to adjust the f2pts to segmentation observation,
                it might be used to process the source information.

            get_uv_info (bool): get the information for UV, it might be used to process the source information;

        Returns:
            smpl_info (dict):
                --cam   (torch.Tensor): (bs * nt, 3),
                --verts (torch.Tensor): (bs * nt, 6890, 3),
                --j2d   (torch.Tensor): (bs * nt, 19, 2),
                --cond  (torch.Tensor): (bs * nt, 3, h, w),
                --fim   (torch.Tensor): (bs * nt, h, w),
                --wim   (torch.Tensor): (bs * nt, h, w, 3),
                --f2pts (torch.Tensor): (bs * nt, 13776, 3, 2)
                --only_vis_f2pts (torch.tensor): (bs * nt, 13776, 3, 2)
                --obj_fim              (torch.tensor): (bs * nt, h, w),
                --obj_wim              (torch.tensor): (bs * nt, h, w, 3),
                --obj_f2pts            (torch.tensor): (bs * nt, 13776, 3, 2)
                --only_vis_obj_f2pts   (torch.tensor): (bs * nt, 13776, 3, 2)
        """

        f2pts, fim, wim = self.render.render_fim_wim(cam=smpl_info["cam"], vertices=smpl_info["verts"], smpl_faces=True)
        cond, _ = self.render.encode_fim(smpl_info["cam"], smpl_info["verts"], fim=fim, transpose=True)

        if use_morph:
            if "masks" in smpl_info:
                human_sil = 1 - smpl_info["masks"]
            else:
                human_sil = 1 - cond[:, -1:]

            smpl_info["confidant_sil"] = morph(human_sil, ks=self._opt.conf_erode_ks, mode="erode")

            smpl_info["outpad_sil"] = morph(
                ((human_sil + 1 - cond[:, -1:]) > 0).float(),
                ks=self._opt.out_dilate_ks, mode="dilate"
            )
            # f2pts = self.make_morph_f2pts(f2pts, fim, smpl_info["human_sil"], erode_ks=0)

        only_vis_f2pts = self.render.get_vis_f2pts(f2pts, fim)
        smpl_info["f2pts"] = f2pts
        smpl_info["only_vis_f2pts"] = only_vis_f2pts
        smpl_info["cond"] = cond
        smpl_info["fim"] = fim
        smpl_info["wim"] = wim

        if get_uv_info:
            obj_f2pts, obj_fim, obj_wim = self.render.render_fim_wim(
                cam=smpl_info["cam"], vertices=smpl_info["verts"], smpl_faces=False)

            # if use_morph:
            #     obj_f2pts = self.make_morph_f2pts(obj_f2pts, obj_fim, smpl_info["human_sil"], erode_ks=0)

            only_vis_obj_f2pts = self.render.get_vis_f2pts(obj_f2pts, obj_fim)
            smpl_info["obj_f2pts"] = obj_f2pts
            smpl_info["only_vis_obj_f2pts"] = only_vis_obj_f2pts
            smpl_info["obj_fim"] = obj_fim
            smpl_info["obj_wim"] = obj_wim

        return smpl_info

    def make_tsf_inputs(self, uv_img, ref_info):
        """

        Args:
            uv_img (torch.tensor): (bs, 3, h, w)
            ref_info (dict): the dict of smpl details, including
                --cam   (torch.tensor): (bs * nt, 3),
                --verts (torch.tensor): (bs * nt, 6890, 3),
                --j2d   (torch.tensor): (bs * nt, 19, 2),
                --cond  (torch.tensor): (bs * nt, 3, h, w),
                --fim   (torch.tensor): (bs * nt, h, w),
                --wim   (torch.tensor): (bs * nt, h, w, 3),
                --f2pts (torch.tensor): (bs * nt, 13776, 3, 2)
                --only_vis_f2pts (torch.tensor): (bs * nt, 13776, 3, 2)
                --obj_fim              (torch.tensor): (bs * nt, h, w),
                --obj_wim              (torch.tensor): (bs * nt, h, w, 3),
                --obj_f2pts            (torch.tensor): (bs * nt, 13776, 3, 2)
                --only_vis_obj_f2pts   (torch.tensor): (bs * nt, 13776, 3, 2)

        Returns:
            tsf_inputs (torch.tensor): (bs, nt, 6, h, w)
        """

        ref_cond = ref_info["cond"]
        ref_fim = ref_info["fim"]
        ref_wim = ref_info["wim"]

        bs, _, h, w = uv_img.shape
        bsxnt = ref_cond.shape[0]
        nt = bsxnt // bs

        # self.f2uvs = (bs, 13776, 3, 3)
        f2uvs = self.f2uvs[0:bsxnt].clone()

        Tuv2t = self.render.cal_bc_transform(f2uvs, ref_fim, ref_wim)

        syn_img = F.grid_sample(uv_img.unsqueeze(1).repeat(1, nt, 1, 1, 1).view(-1, 3, h, w), Tuv2t)

        tsf_inputs = torch.cat([syn_img, ref_cond], dim=1)

        tsf_inputs = tsf_inputs.view(bs, nt, -1, h, w)

        return tsf_inputs

    def make_bg_inputs(self, src_img, src_info):
        # bg input
        src_cond = src_info["cond"]
        if "masks" in src_info:
            bg_mask = src_info["masks"]
        else:
            bg_mask = src_cond[:, -1:, :, :]

        src_bg_mask = morph(bg_mask, ks=self._opt.bg_ks, mode="erode")
        input_G_bg = torch.cat([src_img * src_bg_mask, src_bg_mask], dim=1)

        return input_G_bg

    def make_src_inputs(self, src_img, src_info):
        input_G_src = torch.cat([src_img, src_info["cond"]], dim=1)

        return input_G_src

    def cal_top_k_ids(self, uncertain_points, boundaries_points, top_k=3):
        """

        Args:
            uncertain_points  (torch.Tensor): (n1, 2)
            boundaries_points (torch.Tensor): (n2, 2)
            top_k (int):
        Returns:
            weights (torch.Tensor): (n1, top_k)
            nn_pts  (torch.Tensor): (n1, top_k, 2)
        """

        n1 = uncertain_points.shape[0]
        n2 = boundaries_points.shape[0]

        u_pts = uncertain_points.unsqueeze(dim=1).expand((n1, n2, 2))
        b_pts = boundaries_points.unsqueeze(dim=0).expand((n1, n2, 2))

        dists = torch.sum((u_pts - b_pts) ** 2, dim=-1)

        val, ids = dists.topk(k=top_k, dim=-1, largest=False, sorted=False)
        val = val.float()

        weights = val / torch.sum(val, dim=1, keepdim=True)
        nn_pts = boundaries_points[ids]

        return weights, nn_pts, ids

    def morph_image(self, src_img, weights, uncertain_pts, nn_pts, confidant_sil):
        """

        Args:
            src_img       (torch.Tensor): (3, h, w)
            weights       (torch.Tensor): (n1, top_k)
            uncertain_pts (torch.Tensor): (n1, 2)
            nn_pts        (torch.Tensor): (n1, top_k, 2)
            confidant_sil (torch.Tensor): (1, h, w)

        Returns:
            morph_img (torch.Tensor): (3, h, w)
        """

        n1, top_k = nn_pts.shape[0:2]
        nn_pts = nn_pts.view(n1 * top_k, -1)

        # (3, n1 * top_k)
        src_rgbs = src_img[:, nn_pts[:, 0], nn_pts[:, 1]]

        # (3, n1, top_k)
        src_rgbs = src_rgbs.view(-1, n1, top_k)

        # (n1, 3, top_k)
        src_rgbs = src_rgbs.permute((1, 0, 2))

        # (n1, 3, top_k) * (n1, top_k, 1) = (n1, 3, 1)
        weights.unsqueeze_(dim=-1)
        uncertain_rgbs = torch.matmul(src_rgbs, weights)
        uncertain_rgbs.squeeze_(dim=-1)

        # (3, n1)
        uncertain_rgbs = uncertain_rgbs.permute((1, 0))

        morph_img = src_img * confidant_sil
        morph_img[:, uncertain_pts[:, 0], uncertain_pts[:, 1]] = uncertain_rgbs

        return morph_img

    def make_morph_image(self, src_img, src_info, erode_ks=3, dilate_ks=11):
        """

        Args:
            src_img   (torch.Tensor): (bs * ns, 3, h, w)
            src_info  (dict):
            erode_ks  (int):
            dilate_ks (int):

        Returns:
            all_morph_imgs (torch.cuda.Tensor): (bs * ns, 3, h, w)
        """

        bs = src_img.shape[0]

        if erode_ks > 0:
            confidant_sil = morph(src_info["confidant_sil"], ks=erode_ks, mode="erode")
        else:
            confidant_sil = src_info["confidant_sil"]

        if dilate_ks > 0:
            outpad_sil = morph(src_info["outpad_sil"], ks=dilate_ks, mode="dilate")
        else:
            outpad_sil = src_info["outpad_sil"]

        # outpad_sil = ((confidant_sil + (1 - src_info["cond"][:, -1:])) > 0).float()
        # outpad_sil = morph(outpad_sil, ks=dilate_ks, mode="dilate")

        canny_filter = CannyFilter(device=src_img.device).to(src_img.device)

        blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges = canny_filter(
            confidant_sil, 0.1, 0.9, True)

        uncertain_sil = outpad_sil * (1 - confidant_sil)

        # visualizer.vis_named_img("silhouette", 1 - src_info["masks"])
        # visualizer.vis_named_img("confidant_sil", confidant_sil)
        # visualizer.vis_named_img("outpad_sil", outpad_sil)
        # visualizer.vis_named_img("uncertain_sil", uncertain_sil)

        all_morph_imgs = []
        for i in range(bs):
            boundaries_points = thin_edges[i, 0].nonzero(as_tuple=False)
            uncertain_points = uncertain_sil[i, 0].nonzero(as_tuple=False)

            weights, nn_pts, ids = self.cal_top_k_ids(uncertain_points, boundaries_points)

            morph_img = self.morph_image(src_img[i], weights, uncertain_points, nn_pts, confidant_sil[i])
            all_morph_imgs.append(morph_img)

        all_morph_imgs = torch.stack(all_morph_imgs, dim=0)
        return all_morph_imgs

    def morph_f2pts(self, f2pts, fbc, fim, confidant_mask):
        """

        Args:
            f2pts          (torch.Tensor): (nf, 3, 2)
            fbc            (torch.Tensor): (nf, 2)
            fim            (torch.Tensor): (h, w)
            confidant_mask (torch.Tensor): (h, w)
        Returns:
            morph_f2pts (torch.Tensor): (nf, 3, 2)
        """

        nf = f2pts.shape[0]

        morphed_f2tps = f2pts.clone()

        valid_fids = fim[((fim != -1).float() * confidant_mask) != 0].unique().long()

        check_valid_fids = torch.zeros(nf, dtype=torch.bool)
        check_valid_fids[valid_fids] = True
        invalid_fids = (~check_valid_fids).nonzero(as_tuple=False)
        invalid_fids.squeeze_(1)

        invalid_fbc = fbc[invalid_fids]
        candidate_pts = fbc[valid_fids]

        weights, nn_pts, ids = self.cal_top_k_ids(invalid_fbc, candidate_pts, top_k=1)
        nn_fids = valid_fids[ids[:, 0]]
        morphed_f2tps[invalid_fids] = f2pts[nn_fids]

        # visualizer.vis_named_img("fim", fim[None][None])
        # visualizer.vis_named_img("mask_fim", (fim * confidant_mask)[None][None])

        return morphed_f2tps

    def make_morph_f2pts(self, f2pts, fim, human_sil, erode_ks=0):
        """

        Args:
            f2pts       (torch.Tensor): (bs, nf, 3, 2)
            fim         (torch.Tensor): (bs, h, w)
            human_sil   (torch.Tensor): (bs, 1, h, w)
            erode_ks    (int)

        Returns:
            morphed_f2pts (torch.Tensor): (bs, nf, 3, 2)
        """

        bs = f2pts.shape[0]

        # (bs, 1, h, w)
        if erode_ks > 0:
            human_sil = morph(human_sil, ks=erode_ks, mode="erode")

        fbc = self.render.compute_barycenter(f2pts)
        morphed_f2pts = []
        for i in range(bs):
            m_f2pts = self.morph_f2pts(f2pts[i], fbc[i], fim[i], human_sil[i][0])
            morphed_f2pts.append(m_f2pts)

        morphed_f2pts = torch.stack(morphed_f2pts, dim=0)

        return morphed_f2pts

    def process_source(self, src_img, src_info, primary_ids=None):
        """
            calculate the inputs for bg_net and src_net.
        Args:
            src_img (torch.Tensor): (bs, ns, 3, h, w)
            src_info (dict): the details information of smpl, including:
                --theta (torch.Tensor): (bs * ns, 85),
                --cam   (torch.Tensor):   (bs * ns, 3),
                --pose  (torch.Tensor):  (bs * ns, 72),
                --shape (torch.Tensor): (bs * ns, 10),
                --verts (torch.Tensor): (bs * ns, 6890, 3),
                --j2d   (torch.Tensor):   (bs * ns, 19, 2),
                --j3d   (torch.Tensor):   (bs * ns, 19, 3),
                --masks (torch.Tensor or None): (bs * ns, 1, h, w)

            primary_ids (List[int] or None)   : set the ids of source images as primary.

        Returns:
            uv_img      (torch.tensor):  (bs, 3, h, w)
            input_G_bg  (torch.tensor):  (bs, 1, 4, h, w)
            input_G_src (torch.tensor):  (bs, ns, 6, h, w)
        """

        bs, ns, _, h, w = src_img.shape
        nt = self._opt.time_step

        self.make_uv_setup(bs, self._opt.num_source, nt, src_img.device)

        # 1. make morph image
        # morph_src_img = self.make_morph_image(src_img.view(bs * ns, 3, h, w),
        #                                       src_info, erode_ks=1, dilate_ks=13)

        morph_src_img = self.make_morph_image(src_img.view(bs * ns, 3, h, w),
                                              src_info, erode_ks=0, dilate_ks=0)

        # 2. merge the uv image from source images
        morph_uv_img = self.make_uv_img(morph_src_img.view(bs, ns, 3, h, w), src_info)    # (bs, 3, h, w)

        # 3. make source inputs, (bs * ns, 6, h, w)
        input_G_src = self.make_src_inputs(morph_src_img.view(bs * ns, 3, h, w), src_info)

        # 4. make background inputs, (bs * ns, 4, h, w)
        input_G_bg = self.make_bg_inputs(src_img.view(bs * ns, 3, h, w), src_info)

        input_G_bg = input_G_bg.view(bs, ns, -1, h, w)
        input_G_src = input_G_src.view(bs, ns, -1, h, w)

        if primary_ids is None:
            primary_ids = np.random.choice(ns, 1)

        input_G_bg = input_G_bg[:, primary_ids]

        # T = self.render.cal_bc_transform(src_info["f2pts"], src_info["fim"], src_info["wim"])
        # morph_T = F.grid_sample(morph_src_img, T)
        # visualizer.vis_named_img("src_img", src_img[0])
        # visualizer.vis_named_img("morph_src_img", morph_src_img)
        # visualizer.vis_named_img("morph_uv_img", morph_uv_img)
        # visualizer.vis_named_img("morph_T", morph_T)
        # visualizer.vis_named_img("src_inputs", input_G_src[:, 0, 0:3])

        return morph_uv_img, input_G_bg, input_G_src

    def make_trans_flow(self, bs, ns, nt, src_info, temp_info, ref_info, temporal=True, use_selected_f2pts=False):
        """
            It needs to be mentioned that this function is used for testing/inference phase, and do not use it in
        training phase.

        Args:
            bs (int): the number of batch size
            ns (int): the number of source image
            nt (int): the number of time-step

            src_info (dict): the dict of smpl details, including
                --fim            (torch.Tensor): (bs * ns, h, w),
                --wim            (torch.Tensor): (bs * ns, h, w, 3),
                --f2pts          (torch.Tensor): (bs * ns, 13776, 3, 3) or (bs * ns, 13776, 3, 2)
                --selected_f2pts (torch.Tensor): (bs * ns, 13776, 3, 3) or (bs * ns, 13776, 3, 2)
                --only_vis_f2pts (torch.Tensor): (bs * ns, 13776, 3, 3) or (bs * ns, 13776, 3, 2)

            temp_info (dict or None): the dict of smpl details, including
                --fim   (torch.Tensor):   (bs * nt, h, w),
                --wim   (torch.Tensor):   (bs * nt, h, w, 3),
                --f2pts (torch.Tensor): (bs * nt, 13776, 3, 3) or (bs * nt, 13776, 3, 2)

            ref_info (dict): the dict of smpl details, including
                --fim   (torch.Tensor):   (bs, h, w),
                --wim   (torch.Tensor):   (bs, h, w, 3),
                --f2pts (torch.Tensor): (bs, 13776, 3, 3) or (bs, 13776, 3, 2)

            temporal (bool): calculate the temporal transformation flow or not, if true and it will return Ttt.
            use_selected_f2pts (bool): use the selected parts or not.

        Returns:
            Tst (torch.tensor):         (bs, nt, ns, h, w, 2),
            Ttt (torch.tensor or None): (bs, nt,
        """
        h, w = self._opt.image_size, self._opt.image_size
        max_ns_nt = max(ns, nt)

        ref_fim_repeat = ref_info["fim"].repeat(max_ns_nt, 1, 1)
        ref_wim_repeat = ref_info["wim"].repeat(max_ns_nt, 1, 1, 1)

        # print("info", bs, ns, nt, ref_fim_repeat.shape, ref_wim_repeat.shape)

        if use_selected_f2pts:
            src_f2pts = src_info["selected_f2pts"]
        else:
            if self._opt.only_vis:
                src_f2pts = src_info["only_vis_f2pts"]
            else:
                src_f2pts = src_info["f2pts"]

        # 1. source to reference transformation flow
        ref_fim = ref_fim_repeat[0:ns * bs]
        ref_wim = ref_wim_repeat[0:ns * bs]
        Tst = self.render.cal_bc_transform(src_f2pts, ref_fim, ref_wim).view(bs, ns, h, w, 2)

        # 2. temporal to reference transformation flow
        if temporal:
            if nt == max_ns_nt:
                ref_fim = ref_fim_repeat
                ref_wim = ref_wim_repeat
            else:
                ref_fim = ref_fim_repeat[0:nt * bs]
                ref_wim = ref_wim_repeat[0:nt * bs]
            # print(temp_info["f2verts"].shape, ref_fim.shape, ref_wim.shape)
            Ttt = self.render.cal_bc_transform(temp_info["f2pts"], ref_fim, ref_wim).view(bs, nt, h, w, 2)
        else:
            Ttt = None

        return Tst, Ttt

    def make_batch_trans_flow(self, bs, ns, nt, src_info, ref_info, temporal=True, use_selected_f2pts=False):
        """
            It needs to be mentioned that this function is used for training phase, and do not use it in
        testing/inference phase.

        Args:
            bs (int): the number of batch size
            ns (int): the number of source image
            nt (int): the number of time-step

            src_info (dict): the dict of smpl details, including
                --fim   (torch.tensor):   (bs * ns, h, w),
                --wim   (torch.tensor):   (bs * ns, h, w, 3),
                --f2verts (torch.tensor): (bs * ns, 13776, 3, 3) or (bs * ns, 13776, 3, 2)

            ref_info (dict): the dict of smpl details, including
                --fim   (torch.tensor):   (bs * nt, h, w),
                --wim   (torch.tensor):   (bs * nt, h, w, 3),
                --f2verts (torch.tensor): (bs * nt, 13776, 3, 3) or (bs * nt, 13776, 3, 2)

            temporal (bool): calculate the temporal transformation flow or not, if true and it will return Ttt.
            use_selected_f2pts (bool): use the selected parts or not.

        Returns:
            Tst (torch.tensor):         (bs, nt, ns, h, w, 2),
            Ttt (torch.tensor or None): (bs, nt,
        """
        h, w = self._opt.image_size, self._opt.image_size
        max_ns_nt = max(ns, nt)

        # (bs * ns, 13776, 3, 2)
        if use_selected_f2pts:
            src_f2pts = src_info["selected_f2pts"]
        else:
            if self._opt.only_vis:
                src_f2pts = src_info["only_vis_f2pts"]
            else:
                src_f2pts = src_info["f2pts"]

        # TODO: how to replace the `repeat` operation with others for memory and computational optimization ?
        # (bs, nt, ns, 13776, 3, 2)
        src_f2pts = src_f2pts.view(bs, ns, 13776, 3, 2).unsqueeze(1)
        src_f2pts = src_f2pts.repeat(1, nt, 1, 1, 1, 1)

        # (bs, nt, max_ns_nt, h, w)
        ref_fim = ref_info["fim"].view(bs, nt, h, w).unsqueeze(2).repeat(1, 1, max_ns_nt, 1, 1)
        # (bs, nt, max_ns_nt, h, w, 3)
        ref_wim = ref_info["wim"].view(bs, nt, h, w, 3).unsqueeze(2).repeat(1, 1, max_ns_nt, 1, 1, 1)

        # print(src_f2pts.shape, ref_fim.shape, ref_wim.shape)

        Tst = self.render.cal_bc_transform(
            src_f2pts.view(-1, 13776, 3, 2),
            ref_fim[:, :, 0:ns].contiguous().view(-1, h, w),
            ref_wim[:, :, 0:ns].contiguous().view(-1, h, w, 3)
        ).view(bs, nt, ns, h, w, 2)

        if temporal:
            # (bs * (nt - 1), 13776, 3, 2) -> (bs, nt - 1, 13776, 3, 2)
            if self._opt.only_vis:
                ref_f2pts = ref_info["only_vis_f2pts"]
            else:
                ref_f2pts = ref_info["f2pts"]

            ref_f2pts = ref_f2pts[0:-bs]

            ref_fim = ref_info["fim"][bs:]
            ref_wim = ref_info["wim"][bs:]

            Ttt = self.render.cal_bc_transform(
                ref_f2pts,
                ref_fim,
                ref_wim
            ).view(bs, nt - 1, h, w, 2)

        else:
            Ttt = None

        return Tst, Ttt

    def forward(self, src_img, ref_img, src_smpl, ref_smpl, src_mask=None, ref_mask=None,
                links_ids=None, offsets=0, temporal=False):
        """
            Make the inputs for bg_net, src_net, and tsf_net. It needs to be mentioned that this function is only
            for training phase, do not use it in testing or inference phase.

        Args:
            src_img (torch.Tensor)         : (bs, ns, 3, H, W)
            ref_img (torch.Tensor)         : (bs, nt, 3, H, W)
            src_smpl (torch.Tensor)        : (bs, ns, 85)
            ref_smpl (torch.Tensor)        : (bs, ns, 85)
            src_mask (torch.Tensor)        : (bs, ns, 3, H, W) or None, front is 0, background is 1
            ref_mask (torch.Tensor)        : (bs, nt, 3, H, W) or None, front is 0, background is 1
            links_ids (torch.Tensor)       : (bs, nv, 3) or (nv, 3) or None
            offsets (torch.Tensor)         : (bs, nv, 3) or 0
            temporal (bool): if true, then it will calculate the temporal warping flow, otherwise Ttt will be None

        Returns:
            input_G_bg  (torch.tensor) :  (bs, ns, 4, H, W)
            input_G_src (torch.tensor) :  (bs, ns, 6, H, W)
            input_G_tsf (torch.tensor) :  (bs, nt, 3 or 6, H, W)
            Tst         (torch.tensor) :  (bs, nt, ns, H, W, 2)
            Ttt         (torch.tensor) :  (bs, nt - 1, nt - 1, H, W, 2) if temporal is True else return None

        """
        bs, ns, _, h, w = src_img.shape
        bs, nt = ref_img.shape[0:2]

        self.make_uv_setup(bs, self._opt.num_source, self._opt.time_step, src_img.device)

        # reshape (view) all into batch-based shape
        if links_ids is not None:
            _, nv, c = links_ids.shape
            src_links_ids = links_ids.expand((bs, ns, nv, c)).view(bs * ns, nv, c)
            ref_links_ids = links_ids.expand((bs, nt, nv, c)).view(bs * nt, nv, c)
        else:
            src_links_ids = None
            ref_links_ids = None

        src_info = self.smpl.get_details(src_smpl.view(bs * ns, -1), offsets, links_ids=src_links_ids)
        ref_info = self.smpl.get_details(ref_smpl.view(bs * nt, -1), offsets, links_ids=ref_links_ids)

        if src_mask is not None:
            src_info["masks"] = src_mask.view((bs * ns, 1, h, w))

        if ref_mask is not None:
            ref_info["masks"] = ref_mask.view((bs * nt, 1, h, w))

        # reshape (view) all into batch-based shape
        self.add_rendered_f2verts_fim_wim(src_info, use_morph=True, get_uv_info=True)
        self.add_rendered_f2verts_fim_wim(ref_info, use_morph=False, get_uv_info=False)

        # The details of source or reference information (src_info / ref_info), including:
        #     --theta (torch.tensor): (bs * ns, 85),
        #     --cam (torch.tensor):   (bs * ns, 3),
        #     --pose (torch.tensor):  (bs * ns, 72),
        #     --shape (torch.tensor): (bs * ns, 10),
        #     --verts (torch.tensor): (bs * ns, 6890, 3),
        #     --j2d (torch.tensor):   (bs * ns, 19, 2),
        #     --j3d (torch.tensor):   (bs * ns, 19, 3)

        # 1. merge the uv image from source images, (bs, 3, h, w)
        # 2. make source inputs, (bs, ns, 6, h, w)
        # 3. make background inputs, (bs, ns, 4, h, w)

        if not self._opt.share_bg:
            primary_ids = np.arange(ns)
        else:
            primary_ids = None

        uv_img, input_G_bg, input_G_src = self.process_source(src_img, src_info, primary_ids=primary_ids)

        # 4. make tsf inputs
        # 4.1 make tsf inputs, (bs, nt, 6, h, w)
        input_G_tsf = self.make_tsf_inputs(uv_img, ref_info)

        # 4.2 calculate transformation flow from sources,  (bs, ns, nt, h, w, 2)
        # 4.3 calculate transformation flow from temporal, (bs, nt, nt - 1, h, w, 2)
        Tst, Ttt = self.make_batch_trans_flow(bs, ns, nt, src_info, ref_info, temporal=self._opt.temporal)

        return input_G_bg, input_G_src, input_G_tsf, Tst, Ttt, uv_img, src_info, ref_info


class FlowCompositionForSwapper(FlowComposition):

    def __init__(self, opt):
        super(FlowCompositionForSwapper, self).__init__(opt)
        self._opt = opt
        self._name = "FlowCompositionForSwapper"

        self.all_faces_ids = list(range(self.render.nf))
        self.part_faces = list(self.render.body_parts.values())

    def make_uv_setup(self, bs, ns, nt, device):
        super(FlowCompositionForSwapper, self).make_uv_setup(bs, ns, nt, device)

        if self.f2uvs is None:
            self.uv_parts = self.render.encode_fim(fim=self.uv_fim, transpose=True)

    def get_selected_fids(self, selected_part_ids):
        """

        Args:
            selected_part_ids (list of int):

        Returns:
            selected_fids (list of int):

        """

        selected_fids = set()
        for i in selected_part_ids:
            fs = set(self.part_faces[i])
            selected_fids |= fs

        selected_fids = list(selected_fids)

        return selected_fids

    def get_select_left_info(self, part_name="body"):
        # get the part ids
        selected_part_ids = self.PART_IDS[part_name]
        left_part_ids = [i for i in self.PART_IDS["all"] if i not in selected_part_ids]

        # get the face ids
        selected_fids = self.get_selected_fids(selected_part_ids)
        left_fids = self.get_selected_fids(left_part_ids)

        return selected_part_ids, left_part_ids, selected_fids, left_fids

    def add_rendered_selected_f2pts(self, src_info, selected_fids):
        """

        Args:
            src_info (dict):
            selected_fids (List[List[int]]):
        Returns:

        """
        obj_f2pts = src_info["obj_f2pts"]
        selected_obj_f2pts = self.render.get_selected_f2pts(obj_f2pts, selected_fids)
        src_info["selected_obj_f2pts"] = selected_obj_f2pts

        f2pts = src_info["f2pts"]
        selected_f2pts = self.render.get_selected_f2pts(f2pts, selected_fids)
        src_info["selected_f2pts"] = selected_f2pts

        if self._opt.only_vis:
            fim = src_info["fim"]
            src_info["selected_obj_f2pts"] = self.render.get_vis_f2pts(selected_obj_f2pts, fim)
            src_info["selected_f2pts"] = self.render.get_vis_f2pts(selected_f2pts, fim)

    def merge_uv_img(self, src_info_list):
        """
            [src_img_0, src_img_1, ..., src_img_sn-1], src_img[0: primary + 1] are the primary images.
            for the selected area of the primary, average the area of the primary and the left;
            for the selected are of the left, only uses the selected area.
            This operation is going to reduce the contrast lighting among the sources.

        Args:
            src_info_list (List[dict]):

        Returns:
            merge_uv (torch.tensor): (bs, 3, h, w)
        """

        merge_uv_img = []
        merge_uv_vis = []

        for i, src_info in enumerate(src_info_list):
            uv_img = src_info["uv_img"]
            selected_src_f2pts = src_info["selected_obj_f2pts"][0:1]
            Ts2uv = self.render.cal_bc_transform(selected_src_f2pts, self.uv_fim[0:1], self.uv_wim[0:1])
            vis_warp_to_uv = F.grid_sample(self.one_map[0:1], Ts2uv)

            merge_uv_img.append(uv_img)
            merge_uv_vis.append(vis_warp_to_uv)

        merge_uv_img = torch.cat(merge_uv_img, dim=0)
        merge_uv_vis = torch.cat(merge_uv_vis, dim=0)
        norm_uv_vis = merge_uv_vis / (torch.sum(merge_uv_vis, dim=0, keepdim=True) + 1e-7)
        uv_img = torch.sum(merge_uv_img * norm_uv_vis, dim=0, keepdim=True)

        # print(uv_img.shape, norm_uv_vis.max(), norm_uv_vis.min())
        #
        # visualizer.vis_named_img(f"merge_uv_img", merge_uv_img)
        # visualizer.vis_named_img(f"merge_uv_vis", merge_uv_vis)
        # visualizer.vis_named_img(f"norm_uv_vis", norm_uv_vis, denormalize=False)
        # visualizer.vis_named_img(f"uv_img", uv_img)

        return uv_img

    def merge_src_info(self, src_info_list, primary_ids):
        """

        Args:
            src_info_list (list of dict):
            primary_ids (int):

        Returns:
            merge_src_info (dict): the source setup information, it contains the followings:
                --cam       (torch.Tensor):             (ns, 3);
                --shape     (torch.Tensor):             (ns, 10);
                --pose      (torch.Tensor):             (ns, 72);
                --fim       (torch.Tensor):             (1 * ns, h, w),
                --wim       (torch.Tensor):             (1 * ns, h, w, 3),
                --f2pts     (torch.Tensor):             (1 * ns, 13776, 3, 2)
                --obj_f2pts (torch.Tensor):             (1 * ns, 13376, 3, 2)
                --selected_f2pts     (torch.Tensor):    (1 * ns, 13776, 3, 2)
                --selected_obj_f2pts (torch.Tensor):    (1 * ns, 13776, 3, 2)
                --only_vis_f2pts     (torch.Tensor):    (1 * ns, 13776, 3, 2)
                --feats     (Tuple[List]): ([(ns, c1, h1, w2), ..., (ns, ck, hk, wk)],
                                            [(ns, ck, hk, wk), ..., (ns, ck, hk, wk)])

                --offsets   (torch.Tensor or 0):    (num_verts, 3) or 0;
                --links_ids (torch.Tensor or None): (num_verts, 3) or None;
                --uv_img    (torch.Tensor):         (1, 3, h, w);
                --bg        (torch.Tensor):         (1, 3, h, w);
        """

        merge_src_info = {
            "img": [],
            "cam": [],
            "shape": [],
            "pose": [],
            "fim": [],
            "wim": [],
            "f2pts": [],
            "obj_f2pts": [],
            "selected_f2pts": [],
            "selected_obj_f2pts": [],
            "only_vis_f2pts": [],
            "feats": ([], []),

            "offsets": [],
            "links_ids": [],
            "uv_img": [],
            "bg_img": [],

            "num_source": 0
        }

        feats_1_list = []
        feats_2_list = []
        for i, src_info in enumerate(src_info_list):

            merge_src_info["num_source"] += src_info["num_source"]

            merge_src_info["img"].append(src_info["img"])
            merge_src_info["cam"].append(src_info["cam"])
            merge_src_info["shape"].append(src_info["shape"])
            merge_src_info["pose"].append(src_info["pose"])
            merge_src_info["fim"].append(src_info["fim"])
            merge_src_info["wim"].append(src_info["wim"])
            merge_src_info["f2pts"].append(src_info["f2pts"])
            merge_src_info["obj_f2pts"].append(src_info["obj_f2pts"])
            merge_src_info["selected_f2pts"].append(src_info["selected_f2pts"])
            merge_src_info["selected_obj_f2pts"].append(src_info["selected_obj_f2pts"])
            merge_src_info["only_vis_f2pts"].append(src_info["only_vis_f2pts"])

            feats_1, feats_2 = src_info["feats"]
            feats_1_list.append(feats_1)
            feats_2_list.append(feats_2)

            if i == primary_ids:
                merge_src_info["offsets"] = src_info["offsets"]
                merge_src_info["links_ids"] = src_info["links_ids"]
                merge_src_info["uv_img"] = src_info["uv_img"]
                merge_src_info["bg"] = src_info["bg"]

        merge_src_info["img"] = torch.cat(merge_src_info["img"], dim=1)
        merge_src_info["cam"] = torch.cat(merge_src_info["cam"], dim=0)
        merge_src_info["shape"] = torch.cat(merge_src_info["shape"], dim=0)
        merge_src_info["pose"] = torch.cat(merge_src_info["pose"], dim=0)
        merge_src_info["fim"] = torch.cat(merge_src_info["fim"], dim=0)
        merge_src_info["wim"] = torch.cat(merge_src_info["wim"], dim=0)
        merge_src_info["f2pts"] = torch.cat(merge_src_info["f2pts"], dim=0)
        merge_src_info["obj_f2pts"] = torch.cat(merge_src_info["obj_f2pts"], dim=0)
        merge_src_info["selected_f2pts"] = torch.cat(merge_src_info["selected_f2pts"], dim=0)
        merge_src_info["selected_obj_f2pts"] = torch.cat(merge_src_info["selected_obj_f2pts"], dim=0)
        merge_src_info["only_vis_f2pts"] = torch.cat(merge_src_info["only_vis_f2pts"], dim=0)

        merge_feats_1 = list(zip(*feats_1_list))
        merge_feats_2 = list(zip(*feats_2_list))

        for merge_feats in merge_feats_1:
            merge_src_info["feats"][0].append(torch.cat(merge_feats, dim=0))

        for merge_feats in merge_feats_2:
            merge_src_info["feats"][1].append(torch.cat(merge_feats, dim=0))

        merge_uv = self.merge_uv_img(src_info_list)

        merge_src_info["uv_img"] = merge_uv

        return merge_src_info
