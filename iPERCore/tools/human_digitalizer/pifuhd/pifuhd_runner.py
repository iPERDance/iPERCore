# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 4:05 下午
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn
# @Editor  : Wen Liu, liuwen@shanghaitech.edu.cn
# @See     : tools/tests/pifuhd_utils_test.py

import shutil
import os
import time
import trimesh


from .core.apps.recon import reconWrapper


class PifuHDRunner(object):
    def __init__(self, ckpt_path="./assets/checkpoints/pose3d/pifuhd.pt",
                 temp_dir="./assets/temp"):
        self.resolution = 512
        self.ckpt_path = ckpt_path
        self.temp_dir = temp_dir

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    @staticmethod
    def mesh_cleaning(obj_path):
        mesh = trimesh.load(obj_path)
        cc = mesh.split(only_watertight=False)

        out_mesh = cc[0]
        bbox = out_mesh.bounds
        height = bbox[1, 0] - bbox[0, 0]
        for c in cc:
            bbox = c.bounds
            if height < bbox[1, 0] - bbox[0, 0]:
                height = bbox[1, 0] - bbox[0, 0]
                out_mesh = c

        out_mesh.export(obj_path)

    def recons(self, cropped_img_path, output_dir, clean_mesh=True):
        """

        Args:
            cropped_img_path (str): the cropped image path;
            output_dir (str): the output dir for the mesh obj;
            clean_mesh (bool): use clean mesh or not;

        Returns:
            None
        """

        dir_name = str(time.time())
        img_name = os.path.split(cropped_img_path)[-1].split(".")[0]
        img_dir = os.path.join(self.temp_dir, dir_name)
        os.makedirs(img_dir, exist_ok=True)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        shutil.copy(cropped_img_path, img_dir)

        cmd = ["--dataroot", img_dir, "--results_path", output_dir,
               "--loadSize", "1024", "--resolution", f"{self.resolution}",
               "--load_netMR_checkpoint_path", self.ckpt_path,
               "--start_id", f"{-1}", "--end_id", f"{-1}"]

        reconWrapper(cmd, use_rect=False, use_cropped=True)

        obj_path = os.path.join(output_dir, img_name + ".obj")

        if clean_mesh:
            self.mesh_cleaning(obj_path)

        shutil.rmtree(img_dir)

        return obj_path
