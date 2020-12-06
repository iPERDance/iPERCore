import torch
import numpy as np
from tqdm import tqdm
from typing import Union, List, Tuple, Any, Dict
from easydict import EasyDict

from .dataset import preprocess, InferenceDataset, InferenceDatasetWithKeypoints
from .network import build_spin
from .. import BasePose3dRunner, BasePose3dRefiner, ACTIONS

from iPERCore.tools.human_digitalizer.bodynets import SMPL
from iPERCore.tools.utils.dataloaders import build_inference_loader
from iPERCore.tools.utils.geometry.boxes import cal_head_bbox
from iPERCore.tools.utils.geometry.cam_pose_utils import cam_init2orig, cam_norm
from iPERCore.tools.utils.filesio.persistence import load_toml_file


__all__ = ["SPINRunner"]


class SPINRunner(BasePose3dRunner):

    def __init__(self,
                 cfg_or_path: Union[EasyDict, str],
                 device=torch.device("cpu")):
        """

        Args:
            cfg_or_path (EasyDict or str): the configuration EasyDict or the cfg_path with `toml` file.
                If it is an EasyDict instance, it must contains the followings,
                    --ckpt_path (str): the path of the pre-trained checkpoints;
                    --smpl_path (str): the path of the smpl model;
                    --smpl_mean_params (str): the path of the mean parameters of SMPL.

                Otherwise if it is a `toml` file, an example could be the followings,
                    ckpt_path = "./assets/pretrains/spin_ckpt.pth"
                    smpl_path = "./assets/pretrains/smpl_model.pkl"
                    smpl_mean_params = "./assets/pretrains/smpl_mean_params.npz"

            device (torch.device):
        """

        self.device = device

        # RGB
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406])[None, :, None, None].to(self.device)
        self.STD = torch.as_tensor([0.229, 0.224, 0.225])[None, :, None, None].to(self.device)

        if isinstance(cfg_or_path, str):
            cfg = EasyDict(load_toml_file(cfg_or_path))
        else:
            cfg = cfg_or_path

        self.model = build_spin(pretrained=False)
        checkpoint = torch.load(cfg["ckpt_path"])
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()

        self._smpl = SMPL(cfg["smpl_path"]).to(self.device)
        self.model = self.model.to(self.device)

    def __call__(self, image: np.ndarray,
                 boxes: Union[np.ndarray, List, Tuple, Any],
                 action: ACTIONS = ACTIONS.SPLIT) -> Dict[str, Any]:
        """

        Args:
            image (np.ndarray): (H, W, C), color intensity [0, 255] with BGR color channel;
            boxes (np.ndarray or List, or Tuple or None): (N, 4)
            action:
                -- 0: only return `cams`, `pose` and `shape` of SMPL;
                -- 1: return `cams`, `pose`, `shape` and `verts`.
                -- 2: return `cams`, `pose`, `shape`, `verts`, `j2d` and `j3d`.

        Returns:
            result (dict):
        """

        image = np.copy(image)
        proc_img, proc_info = preprocess(image, boxes)

        proc_img = torch.tensor(proc_img).to(device=self.device)[None]

        with torch.no_grad():
            proc_img = (proc_img - self.MEAN) / self.STD

            smpls = self.model(proc_img)

            cams_orig = cam_init2orig(smpls[:, 0:3], proc_info["scale"],
                                      torch.tensor(proc_info["start_pt"], device=self.device).float())
            cams = cam_norm(cams_orig, proc_info["im_shape"][0])
            smpls[:, 0:3] = cams

            if action == ACTIONS.SPLIT:
                result = self.body_model.split(smpls)

            elif action == ACTIONS.SKIN:
                result = self.body_model.skinning(smpls)

            elif action == ACTIONS.SMPL:
                result = {"theta": smpls}

            else:
                result = self.body_model.get_details(smpls)

            result["proc_info"] = proc_info

        return result

    def run_with_smplify(self, image_paths: List[str], boxes: List[Union[List, Tuple, np.ndarray]],
                         keypoints_info: Dict, smplify_runner: BasePose3dRefiner,
                         batch_size: int = 16, num_workers: int = 4,
                         filter_invalid: bool = True, temporal: bool = True):
        """

        Args:
            image_paths (list of str): the image paths;
            boxes (list of Union[np.np.ndarray, list, tuple)): the bounding boxes of each image;
            keypoints_info (Dict): the keypoints information of each image;
            smplify_runner (BasePose3dRefiner): the simplify instance, it must contains the keypoint_formater;
            batch_size (int): the mini-batch size;
            num_workers (int): the number of processes;
            filter_invalid (bool): the flag to control whether filter invalid frames or not;
            temporal (bool): use temporal smooth optimization or not.

        Returns:
            smpl_infos (dict): the estimated smpl infomations, it contains,
                --all_init_smpls (torch.Tensor): (num, 85), the initialized smpls;
                --all_opt_smpls (torch.Tensor): (num, 85), the optimized smpls;
                --all_valid_ids (torch.Tensor): (num of valid frames,), the valid indexes.
        """

        def head_is_valid(head_boxes):
            return (head_boxes[:, 1] - head_boxes[:, 0]) * (head_boxes[:, 3] - head_boxes[:, 2]) > 10 * 10

        dataset = InferenceDatasetWithKeypoints(image_paths, boxes, keypoints_info,
                                                smplify_runner.keypoint_formater, image_size=224, temporal=temporal)

        data_loader = build_inference_loader(dataset, batch_size=batch_size, num_workers=num_workers)

        """
        sample (dict): the sample information, it contains,
              --image (torch.Tensor): (3,  224, 224) is the cropped image range of [0, 1] and normalized
                    by MEAN and STD, RGB channel;
              --orig_image (torch.Tensor): (3, height, width) is the in rage of [0, 1], RGB channel;
              --im_shape (torch.Tensor): (height, width)
              --keypoints (dict): (num_joints, 3), and num_joints could be [75,].
              --center (torch.Tensor): (2,);
              --start_pt (torch.Tensor): (2,);
              --scale (torch.Tensor): (1,);
              --img_path (str): the image path.
        """

        all_init_smpls = []
        all_opt_smpls = []
        all_pose3d_img_ids = []
        for sample in tqdm(data_loader):
            images = sample["image"].to(self.device)
            start_pt = sample["start_pt"].to(self.device)
            scale = sample["scale"][:, None].to(self.device).float()
            im_shape = sample["im_shape"][:, 0:1].to(self.device)
            keypoints_info = sample["keypoints"].to(self.device)

            img_ids = sample["img_id"]

            with torch.no_grad():
                init_smpls = self.model(images)
            cams_orig = cam_init2orig(init_smpls[:, 0:3], scale, start_pt)
            cams = cam_norm(cams_orig, im_shape)
            init_smpls[:, 0:3] = cams

            smplify_results = smplify_runner(
                keypoints_info, cams, init_smpls[:, -10:], init_smpls[:, 3:-10], proc_kps=False, temporal=temporal
            )
            opt_smpls = torch.cat([cams, smplify_results["new_opt_pose"], smplify_results["new_opt_betas"]], dim=1)

            if filter_invalid:
                opt_smpls_info = self.get_details(opt_smpls)
                head_boxes = cal_head_bbox(opt_smpls_info["j2d"], image_size=512)
                valid = head_is_valid(head_boxes).nonzero(as_tuple=False)
                valid.squeeze_(-1)
                img_ids = img_ids[valid]

            all_init_smpls.append(init_smpls.cpu())
            all_opt_smpls.append(opt_smpls.cpu())
            all_pose3d_img_ids.append(img_ids.cpu())

        all_init_smpls = torch.cat(all_init_smpls, dim=0)
        all_opt_smpls = torch.cat(all_opt_smpls, dim=0)
        all_valid_ids = torch.cat(all_pose3d_img_ids, dim=0)

        smpl_infos = {
            "all_init_smpls": all_init_smpls,
            "all_opt_smpls": all_opt_smpls,
            "all_valid_ids": all_valid_ids
        }

        return smpl_infos

    def run(self, image_paths: List[str], boxes: List[List],
            batch_size: int = 16, num_workers: int = 4,
            filter_invalid: bool = True, temporal: bool = True):

        """

        Args:
            image_paths (list of str): the image paths;
            boxes (list of list): the bounding boxes of each image;
            batch_size (int): the mini-batch size;
            num_workers (int): the number of processes;
            filter_invalid (bool): the flag to control whether filter invalid frames or not;
            temporal (bool): use temporal smooth optimization or not.

        Returns:
            smpl_infos (dict): the estimated smpl infomations, it contains,
                --all_init_smpls (torch.Tensor): (num, 85), the initialized smpls;
                --all_opt_smpls (torch.Tensor): None
                --all_valid_ids (torch.Tensor): (num of valid frames,), the valid indexes.
        """

        def head_is_valid(head_boxes):
            return (head_boxes[:, 1] - head_boxes[:, 0]) * (head_boxes[:, 3] - head_boxes[:, 2]) > 10 * 10

        dataset = InferenceDataset(image_paths, boxes, image_size=224)
        data_loader = build_inference_loader(dataset, batch_size=batch_size, num_workers=num_workers)

        """
        sample (dict): the sample information, it contains,
              --image (torch.Tensor): (3,  224, 224) is the cropped image range of [0, 1] and normalized
                    by MEAN and STD, RGB channel;
              --orig_image (torch.Tensor): (3, height, width) is the in rage of [0, 1], RGB channel;
              --im_shape (torch.Tensor): (height, width)
              --keypoints (dict): (num_joints, 3), and num_joints could be [75,].
              --center (torch.Tensor): (2,);
              --start_pt (torch.Tensor): (2,);
              --scale (torch.Tensor): (1,);
              --img_path (str): the image path.
        """

        all_init_smpls = []
        all_pose3d_img_ids = []
        for sample in tqdm(data_loader):
            images = sample["image"].to(self.device)
            start_pt = sample["start_pt"].to(self.device)
            scale = sample["scale"][:, None].to(self.device).float()
            im_shape = sample["im_shape"][:, 0:1].to(self.device)
            img_ids = sample["img_id"]

            with torch.no_grad():
                init_smpls = self.model(images)
            cams_orig = cam_init2orig(init_smpls[:, 0:3], scale, start_pt)
            cams = cam_norm(cams_orig, im_shape)
            init_smpls[:, 0:3] = cams

            if filter_invalid:
                init_smpls_info = self.get_details(init_smpls)
                head_boxes = cal_head_bbox(init_smpls_info["j2d"], image_size=512)
                valid = head_is_valid(head_boxes).nonzero(as_tuple=False)
                valid.squeeze_(-1)
                img_ids = img_ids[valid]

            all_init_smpls.append(init_smpls.cpu())
            all_pose3d_img_ids.append(img_ids.cpu())

        all_init_smpls = torch.cat(all_init_smpls, dim=0)
        all_valid_ids = torch.cat(all_pose3d_img_ids, dim=0)

        smpl_infos = {
            "all_init_smpls": all_init_smpls,
            "all_opt_smpls": None,
            "all_valid_ids": all_valid_ids
        }

        return smpl_infos

    def get_details(self, smpls):
        return self._smpl.get_details(smpls)

    @property
    def mean_theta(self):
        mean_cam = self.model.init_cam
        mean_pose = self.model.init_pose
        mean_shape = self.model.init_shape

        mean_theta = torch.cat([mean_cam, mean_pose, mean_shape], dim=-1)[0]
        return mean_theta

    @property
    def body_model(self):
        return self._smpl
