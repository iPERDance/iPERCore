# The modification of https://github.com/vchoutas/smplx/blob/master/smplx/body_models.py

import os
import os.path as osp
import torch
import pickle
import numpy as np

from .base_smpl import BaseSMPL

from iPERCore.tools.human_digitalizer.smplx import SMPL as SMPLX_SMPL
from iPERCore.tools.human_digitalizer.smplx.vertex_ids import vertex_ids as VERTEX_IDS
from iPERCore.tools.human_digitalizer.smplx.utils import Struct
from iPERCore.tools.human_digitalizer.smplx.lbs import lbs


class SMPLH(SMPLX_SMPL, BaseSMPL):
    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = SMPLX_SMPL.NUM_JOINTS - 2
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS

    def __init__(
        self, model_path,
        use_pca: bool = False,
        num_pca_comps: int = 6,
        gender: str = "neutral",
        dtype=torch.float32,
        vertex_ids=None,
        ext: str = "pkl",
        **kwargs
    ) -> None:
        """ SMPLH model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_left_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the left
                hand. (default = True)
            left_hand_pose: torch.tensor, optional, BxP
                The default value for the left hand pose member variable.
                (default = None)
            create_right_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the right
                hand. (default = True)
            right_hand_pose: torch.tensor, optional, BxP
                The default value for the right hand pose member variable.
                (default = None)
            num_pca_comps: int, optional
                The number of PCA components to use for each hand.
                (default = 6)
            flat_hand_mean: bool, optional
                If False, then the pose of the hand is initialized to False.
            batch_size: int, optional
                The batch size used for creating the member variables
            gender: str, optional
                Which gender to load
            dtype: torch.dtype, optional
                The data type for the created variables
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        """

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps

        """
        load the model
        """
        if osp.isdir(model_path):
            model_fn = "SMPLH_{}.{ext}".format(gender.upper(), ext=ext)
            smplh_path = os.path.join(model_path, model_fn)
        else:
            smplh_path = model_path
        assert osp.exists(smplh_path), "Path {} does not exist!".format(
            smplh_path)

        if ext == "pkl":
            with open(smplh_path, "rb") as smplh_file:
                model_data = pickle.load(smplh_file, encoding="latin1")
        elif ext == "npz":
            model_data = np.load(smplh_path, allow_pickle=True)
        else:
            raise ValueError("Unknown extension: {}".format(ext))
        data_struct = Struct(**model_data)

        if vertex_ids is None:
            vertex_ids = VERTEX_IDS["smplh"]

        SMPLX_SMPL.__init__(self, model_path=model_path,
                            data_struct=data_struct,
                            batch_size=1, vertex_ids=vertex_ids, gender=gender,
                            dtype=dtype, ext=ext, **kwargs)

        self.num_pca_comps = num_pca_comps

        left_hand_components = data_struct.hands_componentsl[:num_pca_comps]
        right_hand_components = data_struct.hands_componentsr[:num_pca_comps]

        self.np_left_hand_components = left_hand_components
        self.np_right_hand_components = right_hand_components

        self.np_hands_meanl = data_struct.hands_meanl.astype(np.float32)
        self.np_hands_meanr = data_struct.hands_meanr.astype(np.float32)

        self.register_buffer(
            "hands_meanl",
            torch.tensor(data_struct.hands_meanl, dtype=dtype))
        self.register_buffer(
            "hands_meanr",
            torch.tensor(data_struct.hands_meanr, dtype=dtype))

        self.register_buffer(
            "hands_mean",
            torch.tensor(self.np_hands_mean, dtype=dtype)
        )

        self.register_buffer(
            "left_hand_components",
            torch.tensor(left_hand_components, dtype=dtype))
        self.register_buffer(
            "right_hand_components",
            torch.tensor(right_hand_components, dtype=dtype))

    @property
    def np_hands_mean(self):
        return np.concatenate([self.np_hands_meanl, self.np_hands_meanr], axis=0)

    def forward(self, beta, theta, offsets=0, links_ids=None, get_skin=False):
        """

        Args:
            beta (torch.Tensor): (batch_size, 10)
            theta (torch.Tensor):
                full pose: (batch_size, (1(global) + 21(body) + 15(left hand) + 15(right hand))*3 --> 52*3 --> 156)
                pca: (batch_size, (1(global) + 21(body))*3 + 6(left hand) + 6(right hand) --> 22*3+6+6 --> 78)

            offsets (torch.Tensor): (batch_size, 6890, 3)
            links_ids None or list of np.ndarray): (from_verts_idx, to_verts_idx)
            get_skin (bool): return skin or not

        Returns:

        """

        bs = theta.shape[0]

        if theta.shape[1] == 72:
            hands_mean = self.hands_mean.repeat(bs, 1)
            theta = torch.cat([theta[:, 0: 66], hands_mean], dim=1)

        if self.use_pca:
            left_hand_pose_pca = theta[:, -12:-6]
            right_hand_pose_pca = theta[:, -6:]
            left_hand_pose = torch.einsum(
                "bi,ij->bj", [left_hand_pose_pca, self.left_hand_components])
            right_hand_pose = torch.einsum(
                "bi,ij->bj", [right_hand_pose_pca, self.right_hand_components])

            full_pose = torch.cat([theta[:, :-12], left_hand_pose, right_hand_pose], dim=1)
        else:
            full_pose = theta

        vertices, joints = lbs(beta, full_pose, self.v_template + offsets,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=True)

        if links_ids is not None:
            vertices = self.link(vertices, links_ids)

        return vertices, joints, full_pose
