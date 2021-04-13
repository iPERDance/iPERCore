# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import neural_renderer as nr

from iPERCore.tools.utils.geometry import mesh

COLORS = {
    "pink": [.7, .7, .9],
    "purple": [.9, .7, .7],
    "cyan": [.7, .75, .5],
    "red": [1.0, 0.0, 0.0],

    "green": [.0, 1., .0],
    "yellow": [1., 1., 0],
    "brown": [.5, .7, .7],
    "blue": [.0, .0, 1.],

    "offwhite": [.8, .9, .9],
    "orange": [.5, .65, .9],

    "grey": [.7, .7, .7],
    "black": [0.0, 0.0, 0.0],
    "white": [1.0, 1.0, 1.0],

    "yellowg": [0.83, 1, 0],
}


def orthographic_proj_withz_idrot(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 3: [sc, tx, ty]
    No rotation!
    Orth preserving the z.
    sc * ( x + [tx; ty])
    as in HMR..
    """
    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    # proj = scale * X
    proj = X

    proj_xy = scale * (proj[:, :, :2] + trans)
    proj_z = proj[:, :, 2, None] + offset_z

    return torch.cat((proj_xy, proj_z), 2)


def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    sc * ( x + [tx; ty])
    as in HMR..
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    # proj = scale * X_rot
    proj = X_rot

    proj_xy = scale * (proj[:, :, :2] + trans)
    proj_z = proj[:, :, 2, None] + offset_z

    return torch.cat((proj_xy, proj_z), 2)


def quat_rotate(X, q):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        q: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]] * 0 + 1
    q = torch.unsqueeze(q, 1) * ones_x

    q_conj = torch.cat([q[:, :, [0]], -1 * q[:, :, 1:4]], dim=-1)
    X = torch.cat([X[:, :, [0]] * 0, X], dim=-1)

    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]


def hamilton_product(qa, qb):
    """Multiply qa by qb.

    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0 * qb_0 - qa_1 * qb_1 - qa_2 * qb_2 - qa_3 * qb_3
    q_mult_1 = qa_0 * qb_1 + qa_1 * qb_0 + qa_2 * qb_3 - qa_3 * qb_2
    q_mult_2 = qa_0 * qb_2 - qa_1 * qb_3 + qa_2 * qb_0 + qa_3 * qb_1
    q_mult_3 = qa_0 * qb_3 + qa_1 * qb_2 - qa_2 * qb_1 + qa_3 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


class BaseSMPLRenderer(nn.Module):
    def __init__(self, face_path="assets/checkpoints/pose3d/smpl_faces.npy",
                 fim_enc_path="assets/configs/pose3d/mapper_fim_enc.txt",
                 uv_map_path="assets/configs/pose3d/mapper_uv.txt",
                 part_path="assets/configs/pose3d/smpl_part_info.json",
                 front_path="assets/configs/pose3d/front_body.json",
                 head_path="assets/configs/pose3d/head.json",
                 facial_path="assets/configs/pose3d/front_facial.json",
                 map_name="uv_seg", tex_size=3, image_size=256,
                 anti_aliasing=True, fill_back=False, background_color=(0, 0, 0),
                 viewing_angle=30, near=0.1, far=25.0,
                 has_front=False, top_k=5):
        """

        Args:
            face_path:
            fim_enc_path:
            uv_map_path:
            part_path:
            map_name:
            tex_size:
            image_size:
            anti_aliasing:
            fill_back:
            background_color:
            viewing_angle:
            near:
            far:
            has_front:
            top_k:
        """

        super(BaseSMPLRenderer, self).__init__()

        self.background_color = background_color
        self.anti_aliasing = anti_aliasing
        self.image_size = image_size
        self.fill_back = fill_back
        self.map_name = map_name

        self.obj_info = mesh.load_obj(fim_enc_path)
        obj_faces = self.obj_info["faces"]
        smpl_faces = np.load(face_path)
        self.base_nf = smpl_faces.shape[0]

        # fill back
        if self.fill_back:
            smpl_faces = np.concatenate((smpl_faces, smpl_faces[:, ::-1]), axis=0)
            obj_faces = np.concatenate((obj_faces, obj_faces[:, ::-1]), axis=0)

        self.nf = smpl_faces.shape[0]
        self.register_buffer("smpl_faces", torch.tensor(smpl_faces.astype(np.int32)).int())
        self.register_buffer("obj_faces", torch.tensor(obj_faces.astype(np.int32)).int())

        map_fn = torch.tensor(mesh.create_mapping(
            map_name, fim_enc_path, contain_bg=True, fill_back=fill_back)).float()
        self.register_buffer("map_fn", map_fn)

        if has_front:
            front_map_fn = torch.tensor(mesh.create_mapping(
                "head", fim_enc_path, part_path=part_path, front_path=front_path, facial_path=facial_path,
                head_path=head_path, contain_bg=True, fill_back=fill_back)).float()
            self.register_buffer("front_map_fn", front_map_fn)
        else:
            self.front_map_fn = None

        self.body_parts = mesh.get_part_ids(
            self.nf, part_info=part_path, fill_back=fill_back)

        f_img2uvs = mesh.get_f2vts(self.obj_info, fill_back=fill_back, z=1)
        face_k_nearest = mesh.find_part_k_nearest_faces(f_img2uvs, self.body_parts, k=top_k)
        self.register_buffer("f_img2uvs", torch.tensor(f_img2uvs).float())
        self.register_buffer("face_k_nearest", torch.tensor(face_k_nearest).long())

        f_uvs2img = mesh.get_f2vts(uv_map_path, fill_back=fill_back, z=1)
        f_uvs2img = f_uvs2img[:, :, 0:2]
        self.register_buffer("f_uvs2img", torch.tensor(f_uvs2img).float())

        self.tex_size = tex_size
        self.register_buffer("coords", self.create_coords(tex_size))
        # (nf, T*T, 2)
        img2uv_sampler = mesh.create_uvsampler(uv_map_path, tex_size=tex_size, fill_back=fill_back)
        self.register_buffer("img2uv_sampler", torch.tensor(img2uv_sampler).float())

        # light
        self.light_intensity_ambient = 1
        self.light_intensity_directional = 0
        self.light_color_ambient = [1, 1, 1]
        self.light_color_directional = [1, 1, 1]
        self.light_direction = [0, 1, 0]

        self.rasterizer_eps = 1e-3

        # project function and camera
        self.near = near
        self.far = far
        self.proj_func = orthographic_proj_withz_idrot
        self.viewing_angle = viewing_angle
        self.eye = [0, 0, -(1. / np.tan(np.radians(self.viewing_angle)) + 1)]

    def set_ambient_light(self, int_dir=0.3, int_amb=0.7, direction=(1, 0.5, 1)):
        self.light_intensity_directional = int_dir
        self.light_intensity_ambient = int_amb
        if direction is not None:
            self.light_direction = direction

    def set_bgcolor(self, color=(-1, -1, -1)):
        self.background_color = color

    def set_tex_size(self, tex_size):
        del self.coords
        self.coords = self.create_coords(tex_size)

    def set_img_size(self, image_size):
        self.image_size = image_size

    def forward(self, cam, vertices, uv_imgs, dynamic=True, get_fim=False):
        bs = cam.shape[0]
        faces = self.smpl_faces.repeat(bs, 1, 1)

        if dynamic:
            samplers = self.dynamic_sampler(cam, vertices, faces)
        else:
            samplers = self.img2uv_sampler.repeat(bs, 1, 1, 1)

        textures = self.extract_tex(uv_imgs, samplers)

        images, fim = self.render(cam, vertices, textures, faces, get_fim=get_fim)

        if get_fim:
            return images, textures, fim
        else:
            return images, textures

    def render(self, cam, vertices, textures, faces=None, get_fim=False):
        if faces is None:
            bs = cam.shape[0]
            faces = self.smpl_faces.repeat(bs, 1, 1)

        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)

        # TODO: this will replace the textures, so, clone `textures` at first.
        textures = textures.clone()
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(faces, textures, self.image_size, self.anti_aliasing,
                              self.near, self.far, self.rasterizer_eps, self.background_color)
        fim = None
        if get_fim:
            fim = nr.rasterize_face_index_map(faces, image_size=self.image_size, anti_aliasing=False,
                                              near=self.near, far=self.far, eps=self.rasterizer_eps)

        return images, fim

    def render_fim(self, cam, vertices, smpl_faces=True):
        if smpl_faces:
            faces = self.smpl_faces
        else:
            faces = self.obj_faces

        bs = cam.shape[0]
        faces = faces.repeat(bs, 1, 1)

        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        fim = nr.rasterize_face_index_map(faces, self.image_size, False)
        return fim

    def render_fim_wim(self, cam, vertices, smpl_faces=True):
        if smpl_faces:
            faces = self.smpl_faces
        else:
            faces = self.obj_faces

        bs = cam.shape[0]
        faces = faces.repeat(bs, 1, 1)

        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        fim, wim = nr.rasterize_face_index_map_and_weight_map(faces, self.image_size, False)

        f2pts = faces[:, :, :, 0:2]
        f2pts[:, :, :, 1] *= -1

        return f2pts, fim, wim

    def render_uv_fim_wim(self, bs):
        """

        Args:
            bs:

        Returns:

        """

        f_img2uvs = self.f_img2uvs.repeat(bs, 1, 1, 1)
        f_img2uvs[:, :, :, 1] *= -1
        fim, wim = nr.rasterize_face_index_map_and_weight_map(f_img2uvs, self.image_size, False)

        return fim, wim

    def render_depth(self, cam, vertices):
        bs = cam.shape[0]
        faces = self.faces.repeat(bs, 1, 1)
        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1

        # rasterization
        faces = self.vertices_to_faces(proj_verts, faces)
        images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render_silhouettes(self, cam, vertices, faces=None):
        if faces is None:
            bs = cam.shape[0]
            faces = self.smpl_faces.repeat(bs, 1, 1)

        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def encode_fim(self, cam=None, vertices=None, fim=None, transpose=True, map_fn=None):
        assert (cam is not None and vertices is not None) or fim is not None

        if map_fn is not None:
            fim_enc = map_fn[fim.long()]
        else:
            fim_enc = self.map_fn[fim.long()]

        if transpose:
            fim_enc = fim_enc.permute(0, 3, 1, 2)

        return fim_enc, fim

    def encode_front_fim(self, fim, transpose=True):
        fim_enc = self.front_map_fn[fim.long()]
        if transpose:
            fim_enc = fim_enc.permute(0, 3, 1, 2)

        return fim_enc

    def extract_tex_from_image(self, images, cam, vertices):
        bs = images.shape[0]
        faces = self.faces.repeat(bs, 1, 1)

        sampler = self.dynamic_sampler(cam, vertices, faces)  # (bs, nf, T*T, 2)

        tex = self.extract_tex(images, sampler)

        return tex

    def extract_tex_from_uv(self, uv_img):
        """

        Args:
            uv_img: (bs, 3, h, w)

        Returns:

        """
        bs = uv_img.shape[0]
        samplers = self.img2uv_sampler.repeat(bs, 1, 1, 1)
        textures = self.extract_tex(uv_img, samplers)

        return textures

    def extract_tex(self, uv_img, uv_sampler):
        """

        Args:
            uv_img: (bs, 3, h, w)
            uv_sampler: (bs, nf, T*T, 2)

        Returns:

        """

        # (bs, 3, nf, T*T)
        tex = F.grid_sample(uv_img, uv_sampler)
        # (bs, 3, nf, T, T)
        tex = tex.view(-1, 3, self.nf, self.tex_size, self.tex_size)
        # (bs, nf, T, T, 3)
        tex = tex.permute(0, 2, 3, 4, 1)
        # (bs, nf, T, T, T, 3)
        tex = tex.unsqueeze(4).repeat(1, 1, 1, 1, self.tex_size, 1)

        return tex

    def dynamic_sampler(self, cam, vertices, faces):
        # ipdb.set_trace()
        points = self.batch_orth_proj_idrot(cam, vertices)  # (bs, nf, 2)
        faces_points = self.points_to_faces(points, faces)  # (bs, nf, 3, 2)
        # print(faces_points.shape)
        sampler = self.points_to_sampler(self.coords, faces_points)  # (bs, nf, T*T, 2)
        return sampler

    def project_to_image(self, cam, vertices):
        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        # proj_verts[:, :, 1] *= -1
        proj_verts = proj_verts[:, :, 0:2]
        return proj_verts

    def points_to_faces(self, points, faces=None):
        """
        Args:
            points:
            faces:

        Returns:

        """
        bs, nv = points.shape[:2]
        device = points.device

        if faces is None:
            faces = self.faces.repeat(bs, 1, 1)
            # if self.fill_back:
            #     faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        points = points.reshape((bs * nv, 2))
        # pytorch only supports long and byte tensors for indexing
        return points[faces.long()]

    @staticmethod
    def compute_barycenter(f2vts):
        """
        Args:
            f2vts: N x F x 3 x 2

        Returns:
            fbc: N x F x 2
        """

        # Compute alpha, beta (this is the same order as NMR)
        v2 = f2vts[:, :, 2]  # (nf, 2)
        # v0v2 = f2vts[:, :, 0] - f2vts[:, :, 2]  # (nf, 2)
        # v1v2 = f2vts[:, :, 1] - f2vts[:, :, 2]  # (nf, 2)

        v0v2 = f2vts[:, :, 0] - v2  # (nf, 2)
        v1v2 = f2vts[:, :, 1] - v2  # (nf, 2)

        fbc = v2 + 0.5 * v0v2 + 0.5 * v1v2

        return fbc

    @staticmethod
    def batch_orth_proj_idrot(camera, X):
        """
        X is N x num_points x 3
        camera is N x 3
        same as applying orth_proj_idrot to each N
        """

        # TODO check X dim size.
        # X_trans is (N, num_points, 2)
        X_trans = X[:, :, :2] + camera[:, None, 1:]
        # reshape X_trans, (N, num_points * 2)
        # --- * operation, (N, 1) x (N, num_points * 2) -> (N, num_points * 2)
        # ------- reshape, (N, num_points, 2)

        return camera[:, None, 0:1] * X_trans

    @staticmethod
    def points_to_sampler(coords, faces):
        """
        Args:
            coords: [2, T*T]
            faces: [batch size, number of vertices, 3, 2]

        Returns:
            [batch_size, number of vertices, T*T, 2]
        """

        # Compute alpha, beta (this is the same order as NMR)
        nf = faces.shape[1]
        v2 = faces[:, :, 2]  # (bs, nf, 2)
        v0v2 = faces[:, :, 0] - faces[:, :, 2]  # (bs, nf, 2)
        v1v2 = faces[:, :, 1] - faces[:, :, 2]  # (bs, nf, 2)

        # bs x  F x 2 x T*2
        samples = torch.matmul(torch.stack((v0v2, v1v2), dim=-1), coords) + v2.view(-1, nf, 2, 1)
        # bs x F x T*2 x 2 points on the sphere
        samples = samples.permute(0, 1, 3, 2)
        samples = torch.clamp(samples, min=-1.0, max=1.0)
        return samples

    @staticmethod
    def create_coords(tex_size=3):
        """

        Args:
            tex_size (int):

        Returns:
            coords (torch.Tensor): （2, tex_size * tex_size)
        """

        if tex_size == 1:
            step = 1
        else:
            step = 1 / (tex_size - 1)

        alpha_beta = torch.arange(0, 1 + step, step, dtype=torch.float32)
        xv, yv = torch.meshgrid([alpha_beta, alpha_beta])

        coords = torch.stack([xv.flatten(), yv.flatten()], dim=0)

        return coords

    @staticmethod
    def create_meshgrid(image_size):
        """
        Args:
            image_size:

        Returns:
            (image_size, image_size, 2)
        """
        factor = torch.arange(0, image_size, dtype=torch.float32) / (image_size - 1)  # [0, 1]
        factor = (factor - 0.5) * 2
        xv, yv = torch.meshgrid([factor, factor])
        # grid = torch.stack([xv, yv], dim=-1)
        grid = torch.stack([yv, xv], dim=-1)
        return grid

    def get_f_uvs2img(self, bs):
        f_uvs2img = self.f_uvs2img.repeat(bs, 1, 1, 1)
        return f_uvs2img

    def get_selected_f2pts(self, f2pts, selected_fids):
        """

        Args:
            f2pts (torch.tensor): (bs, f, 3, 2) or (bs, f, 3, 3)
            selected_fids (list of list):

        Returns:

        """

        def get_selected(orig_f2pts, face_ids):
            """
            Args:
                orig_f2pts: (f, 3, 2) or (f, 3, 3)
                face_ids (list):

            Returns:
                vis_f2pts: (f, 3, 2)
            """
            vis_f2pts = torch.zeros_like(orig_f2pts) - 2.0
            vis_f2pts[face_ids] = orig_f2pts[face_ids]

            return vis_f2pts

        if f2pts.dim() == 4:
            all_vis_f2pts = []
            bs = f2pts.shape[0]
            for i in range(bs):
                all_vis_f2pts.append(get_selected(f2pts[i], selected_fids[i]))

            all_vis_f2pts = torch.stack(all_vis_f2pts, dim=0)

        else:
            all_vis_f2pts = get_selected(f2pts, selected_fids)

        return all_vis_f2pts

    def get_vis_f2pts(self, f2pts, fims):
        """
        Args:
            f2pts: (bs, f, 3, 2) or (bs, f, 3, 3)
            fims:  (bs, 256, 256)

        Returns:

        """

        def get_vis(orig_f2pts, fim):
            """
            Args:
                orig_f2pts: (f, 3, 2) or (f, 3, 3)
                fim: (256, 256)

            Returns:
                vis_f2pts: (f, 3, 2)
            """
            vis_f2pts = torch.zeros_like(orig_f2pts) - 2.0
            # 0 is -1
            face_ids = fim.unique()[1:].long()
            # vis_f2pts[face_ids] = orig_f2pts[face_ids]

            face_k_nearest_ids = self.face_k_nearest[face_ids].unique()
            # print(face_ids.max(), face_ids.min())
            # print(face_k_nearest_ids.max(), face_k_nearest_ids.min())
            vis_f2pts[face_k_nearest_ids] = orig_f2pts[face_k_nearest_ids]

            return vis_f2pts

        if f2pts.dim() == 4:
            all_vis_f2pts = []
            bs = f2pts.shape[0]
            for i in range(bs):
                all_vis_f2pts.append(get_vis(f2pts[i], fims[i]))

            all_vis_f2pts = torch.stack(all_vis_f2pts, dim=0)

        else:
            all_vis_f2pts = get_vis(f2pts, fims)

        return all_vis_f2pts

    def cal_transform(self, bc_f2pts, src_fim, dst_fim):
        """
        Args:
            bc_f2pts:
            src_fim:
            dst_fim:

        Returns:

        """
        device = bc_f2pts.device
        bs = src_fim.shape[0]
        # T = renderer.init_T.repeat(bs, 1, 1, 1)    # (bs, image_size, image_size, 2)
        T = (torch.zeros(bs, self.image_size, self.image_size, 2, device=device) - 2)
        # 2. calculate occlusion flows, (bs, no, 2)
        dst_ids = dst_fim != -1

        # 3. calculate tgt flows, (bs, nt, 2)

        for i in range(bs):
            Ti = T[i]

            tgt_i = dst_ids[i]

            # (nf, 2)
            tgt_flows = bc_f2pts[i, dst_fim[i, tgt_i].long()]  # (nt, 2)
            Ti[tgt_i] = tgt_flows

        return T

    def cal_bc_transform(self, src_f2pts, dst_fims, dst_wims):
        """
        Args:
            src_f2pts: (bs, 13776, 3, 2)
            dst_fims:  (bs, 256, 256)
            dst_wims:  (bs, 256, 256, 3)
        Returns:

        """
        bs = src_f2pts.shape[0]
        T = -2 * torch.ones((bs, self.image_size * self.image_size, 2), dtype=torch.float32, device=src_f2pts.device)

        # print(src_f2pts.shape, dst_fims.shape, dst_wims.shape)

        for i in range(bs):
            # (13776, 3, 2)
            from_faces_verts_on_img = src_f2pts[i]

            # to_face_index_map
            to_face_index_map = dst_fims[i]

            # to_weight_map
            to_weight_map = dst_wims[i]

            # (256, 256) -> (256*256, )
            to_face_index_map = to_face_index_map.long().reshape(-1)
            # (256, 256, 3) -> (256*256, 3)
            to_weight_map = to_weight_map.reshape(-1, 3)

            to_exist_mask = (to_face_index_map != -1)
            # (exist_face_num,)
            to_exist_face_idx = to_face_index_map[to_exist_mask]
            # (exist_face_num, 3)
            to_exist_face_weights = to_weight_map[to_exist_mask]

            # (exist_face_num, 3, 2) * (exist_face_num, 3) -> sum -> (exist_face_num, 2)
            exist_smpl_T = (from_faces_verts_on_img[to_exist_face_idx] * to_exist_face_weights[:, :, None]).sum(dim=1)
            # (256, 256, 2)
            T[i, to_exist_mask] = exist_smpl_T

        T = T.view(bs, self.image_size, self.image_size, 2)

        # T = torch.clamp(-2, 2)

        return T

    @torch.no_grad()
    def color_textures(self, color="purple"):
        global COLORS
        color_val = torch.tensor(COLORS[color][::-1]).float() * 2 - 1.0
        return torch.ones((self.nf, self.tex_size, self.tex_size, self.tex_size, 3), dtype=torch.float32) * color_val


class SMPLRenderer(BaseSMPLRenderer):
    def __init__(self, face_path="assets/checkpoints/pose3d/smpl_faces.npy",
                 fim_enc_path="assets/configs/pose3d/mapper_fim_enc.txt",
                 uv_map_path="assets/configs/pose3d/mapper_uv.txt",
                 part_path="assets/configs/pose3d/smpl_part_info.json",
                 front_path="assets/configs/pose3d/front_body.json",
                 head_path="assets/configs/pose3d/head.json",
                 facial_path="assets/configs/pose3d/front_facial.json",
                 map_name="uv_seg", tex_size=3, image_size=256,
                 anti_aliasing=True, fill_back=False, background_color=(0, 0, 0),
                 viewing_angle=30, near=0.1, far=25.0,
                 has_front=False, top_k=5):
        """

        Args:
            face_path:
            fim_enc_path:
            uv_map_path:
            part_path:
            map_name:
            tex_size:
            image_size:
            anti_aliasing:
            fill_back:
            background_color:
            viewing_angle:
            near:
            far:
            has_front:
            top_k:
        """

        super(SMPLRenderer, self).__init__(face_path=face_path,
                                           fim_enc_path=fim_enc_path,
                                           uv_map_path=uv_map_path,
                                           part_path=part_path,
                                           front_path=front_path,
                                           head_path=head_path,
                                           facial_path=facial_path,
                                           map_name=map_name, tex_size=tex_size, image_size=image_size,
                                           anti_aliasing=anti_aliasing, fill_back=fill_back,
                                           background_color=background_color,
                                           viewing_angle=viewing_angle, near=near, far=far,
                                           has_front=has_front, top_k=top_k)

    def forward(self, cam, vertices, uv_imgs, dynamic=True, get_fim=False):
        bs = cam.shape[0]

        # TODO, bug of neural render with batch_size = 3, https://github.com/daniilidis-group/neural_renderer/issues/29
        if bs == 3:
            images = []
            textures = []
            fim = []

            for i in range(bs):
                _cam = cam[i:i+1]
                _vertices = vertices[i:i+1]
                _uv_imgs = uv_imgs[i:i+1]

                outs = super(SMPLRenderer, self).forward(_cam, _vertices, _uv_imgs, dynamic=dynamic, get_fim=True)

                images.append(outs[0])
                textures.append(outs[1])
                fim.append(outs[2])

            images = torch.cat(images, dim=0)
            textures = torch.cat(textures, dim=0)

            if get_fim:
                fim = torch.cat(fim, dim=0)
                return images, textures, fim
            else:
                return images, textures

        else:
            return super(SMPLRenderer, self).forward(cam, vertices, uv_imgs, dynamic=dynamic, get_fim=get_fim)

    def render(self, cam, vertices, textures, faces=None, get_fim=False):
        bs = cam.shape[0]

        # TODO, bug of neural render with batch_size = 3, https://github.com/daniilidis-group/neural_renderer/issues/29
        if bs == 3:
            images = []
            fim = []

            for i in range(bs):
                _cam = cam[i:i+1]
                _vertices = vertices[i:i+1]

                _images, _fim = super(SMPLRenderer, self).render(_cam, _vertices, textures, faces, get_fim=get_fim)

                images.append(_images)
                fim.append(_fim)

            images = torch.cat(images, dim=0)

            if get_fim:
                fim = torch.cat(fim, dim=0)
                return images, fim
            else:
                return images, None

        else:
            return super(SMPLRenderer, self).render(cam, vertices, textures, faces, get_fim)

    def render_fim(self, cam, vertices, smpl_faces=True):
        bs = cam.shape[0]

        # TODO, bug of neural render with batch_size = 3, https://github.com/daniilidis-group/neural_renderer/issues/29
        if bs == 3:
            fim = []

            for i in range(bs):
                _cam = cam[i:i+1]
                _vertices = vertices[i:i+1]

                _fim = super(SMPLRenderer, self).render_fim(_cam, _vertices, smpl_faces)

                fim.append(_fim)

            fim = torch.cat(fim, dim=0)
            return fim

        else:
            return super(SMPLRenderer, self).render_fim(cam, vertices, smpl_faces)

    def render_fim_wim(self, cam, vertices, smpl_faces=True):
        bs = cam.shape[0]

        # TODO, bug of neural render with batch_size = 3, https://github.com/daniilidis-group/neural_renderer/issues/29
        if bs == 3:
            f2pts = []
            fim = []
            wim = []

            for i in range(bs):
                _cam = cam[i:i+1]
                _vertices = vertices[i:i+1]

                _f2pts, _fim, _wim = super(SMPLRenderer, self).render_fim_wim(_cam, _vertices, smpl_faces)

                f2pts.append(_f2pts)
                fim.append(_fim)
                wim.append(_wim)

            f2pts = torch.cat(f2pts, dim=0)
            fim = torch.cat(fim, dim=0)
            wim = torch.cat(wim, dim=0)

            return f2pts, fim, wim

        else:
            return super(SMPLRenderer, self).render_fim_wim(cam, vertices, smpl_faces)

    def render_uv_fim_wim(self, bs):
        """

        Args:
            bs:

        Returns:

        """

        # TODO, bug of neural render with batch_size = 3, https://github.com/daniilidis-group/neural_renderer/issues/29
        if bs == 3:
            fim = []
            wim = []
            for i in range(bs):
                _fim, _wim = super(SMPLRenderer, self).render_uv_fim_wim(bs=1)
                fim.append(_fim)
                wim.append(_wim)
            fim = torch.cat(fim, dim=0)
            wim = torch.cat(wim, dim=0)
        else:
            return super(SMPLRenderer, self).render_uv_fim_wim(bs)

        return fim, wim

    def render_depth(self, cam, vertices):
        bs = cam.shape[0]

        # TODO, bug of neural render with batch_size = 3, https://github.com/daniilidis-group/neural_renderer/issues/29
        if bs == 3:
            images = []

            for i in range(bs):
                _cam = cam[i:i+1]
                _vertices = vertices[i:i+1]

                _images = super(SMPLRenderer, self).render_depth(_cam, _vertices)

                images.append(_images)

            images = torch.cat(images, dim=0)

            return images

        else:
            return super(SMPLRenderer, self).render_depth(cam, vertices)

    def render_silhouettes(self, cam, vertices, faces=None):
        bs = cam.shape[0]

        # TODO, bug of neural render with batch_size = 3, https://github.com/daniilidis-group/neural_renderer/issues/29
        if bs == 3:

            images = []
            for i in range(bs):
                _cam = cam[i:i+1]
                _vertices = vertices[i:i+1]

                _images = super(SMPLRenderer, self).render_silhouettes(_cam, _vertices, faces)

                images.append(_images)

            images = torch.cat(images, dim=0)

            return images

        else:
            return super(SMPLRenderer, self).render_silhouettes(cam, vertices, faces)

