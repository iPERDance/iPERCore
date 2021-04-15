""" 
Tensorflow SMPL implementation as batch.
Specify joint types:
'coco': Returns COCO+ 19 joints
'lsp': Returns H3.6M-LSP 14 joints
Note: To get original smpl joints, use self.J_transformed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_smpl import BaseSMPL


VERT_NOSE = 331
VERT_EAR_L = 3485
VERT_EAR_R = 6880
VERT_EYE_L = 2802
VERT_EYE_R = 6262


def load_pickle_file(pkl_path):
    import pickle
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data


def batch_skew(vec, batch_size=None, device="cpu"):
    """
    vec is N x 3, batch_size is int.

    e.g. r = [rx, ry, rz]
        skew(r) = [[ 0,    -rz,      ry],
                   [ rz,     0,     -rx],
                   [-ry,    rx,       0]]

    returns N x 3 x 3. Skew_sym version of each matrix.
    """

    if batch_size is None:
        batch_size = vec.shape[0]

    col_inds = np.array([1, 2, 3, 5, 6, 7], dtype=np.int64)

    # indices = torch.from_numpy(np.reshape(
    #     np.reshape(np.arange(0, batch_size) * 9, [-1, 1]) + col_inds,
    #     newshape=(-1,))).to(device)

    # For better compatibility， since if indices is torch.tensor, it must be long dtype.
    # For fixed index, np.ndarray might be better.
    indices = np.reshape(np.reshape(
        np.arange(0, batch_size) * 9, [-1, 1]) + col_inds, newshape=(-1, )).astype(np.int64)

    updates = torch.stack(
        [
            -vec[:, 2], vec[:, 1], vec[:, 2],
            -vec[:, 0], -vec[:, 1], vec[:, 0]
        ],
        dim=1
    ).view(-1).to(device)

    res = torch.zeros(batch_size * 9, dtype=vec.dtype).to(device)
    res[indices] = updates
    res = res.view(batch_size, 3, 3)

    return res


def batch_rodrigues(theta):
    """
    Theta is N x 3

    rodrigues (from cv2.rodrigues):
    source: https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    input: r (3 x 1)
    output: R (3 x 3)

        angle = norm(r)
        r = r / angle

        skew(r) = [[ 0,    -rz,      ry],
                   [ rz,     0,     -rx],
                   [-ry,    rx,       0]]

        R = cos(theta * eye(3) + (1 - cos(theta)) * r * r.T + sin(theta) *  skew(r)
    """
    batch_size = theta.shape[0]
    device = theta.device

    # angle (batch_size, 1), r (batch_size, 3)
    angle = torch.norm(theta + 1e-8, p=2, dim=1, keepdim=True)
    r = torch.div(theta, angle)

    # angle (batch_size, 1, 1), r (batch_size, 3, 1)
    angle = angle.unsqueeze(-1)
    r = r.unsqueeze(-1)

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    # outer (batch_size, 3, 3)
    outer = torch.matmul(r, r.permute(0, 2, 1))
    eyes = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    R = cos * eyes + (1 - cos) * outer + sin * batch_skew(r, batch_size=batch_size, device=device)

    return R


def batch_rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def batch_lrotmin(theta):
    """ NOTE: not used bc I want to reuse R and this is simple.
    Output of this is used to compute joint-to-pose blend shape mapping.
    Equation 9 in SMPL paper.


    Args:
      pose: `Tensor`, N x 72 vector holding the axis-angle rep of K joints.
            This includes the global rotation so K=24

    Returns
      diff_vec : `Tensor`: N x 207 rotation matrix of 23=(K-1) joints with identity subtracted.,
    """

    # ignore global, N x 72
    theta = theta[:, 3:]
    # (N*23) x 3 x 3
    # reshape = contiguous + view
    Rs = batch_rodrigues(theta.reshape(-1, 3))
    eye = torch.eye(3).to(torch.eye(3))
    lrotmin = (Rs - eye).view(-1, 207)

    return lrotmin


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False, device="cpu"):
    """
    Computes absolute joint locations given pose.

    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.

    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index

    Returns
      new_J : `Tensor`: N x 24 x 3 location of absolute joints
      A     : `Tensor`: N x 24 x 4 x 4 relative joint transformations for LBS.
    """

    N = Rs.shape[0]
    if rotate_base:
        # print('Flipping the SMPL coordinate frame!!!!')
        # rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=Rs.dtype)
        # rot_x = np.reshape(np.tile(rot_x, [N, 1]), (N, 3, 3))
        # root_rotation = np.matmul(Rs[:, 0, :, :], rot_x)

        rot_x = torch.from_numpy(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                                          dtype=np.float32)).type(Rs.dtype).to(device)

        # rot_x = torch.from_numpy(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        #                                   dtype=np.float32)).type(Rs.dtype).to(device)

        rot_x = rot_x.repeat(N, 1).view(N, 3, 3)
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]

    # Now Js is N x 24 x 3 x 1
    Js = Js.unsqueeze(-1)

    def make_A(R, t):
        """
        Composite homogeneous matrix.
        Args:
            R: N x 3 x 3 rotation matrix.
            t: N x 3 x 1 translation vector.

        Returns:
            homogeneous matrix N x 4 x 4.
        """

        # # Rs is N x 3 x 3, ts is N x 3 x 1
        # R_homo = np.pad(R, [[0, 0], [0, 1], [0, 0]], mode='constant')
        # t_homo = np.concatenate([t, np.ones((N, 1, 1))], 1)
        # return np.concatenate([R_homo, t_homo], 2)

        # Pad to (N, 4, 3)
        R_homo = F.pad(R, (0, 0, 0, 1, 0, 0), mode='constant', value=0)
        # Concatenate to (N, 4, 1)
        t_homo = torch.cat([t, torch.ones(N, 1, 1, dtype=Rs.dtype).to(device)], dim=1)
        return torch.cat([R_homo, t_homo], dim=2)

    # root_rotation: (N, 3, 3), Js[:, 0]: (N, 3, 1)
    # ---------- A0: (N, 4, 4)
    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    # N x 24 x 4 x 4
    results = torch.stack(results, dim=1)

    new_J = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---

    # Js_w0: (N, 24, 4, 1)
    Js_w0 = torch.cat([Js, torch.zeros(N, 24, 1, 1, dtype=Rs.dtype).to(device)], dim=2)

    # init_bone: (N, 24, 4, 1) = (N, 24, 4, 4) x (N, 24, 4, 1)
    init_bone = torch.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = F.pad(init_bone, (3, 0, 0, 0, 0, 0, 0, 0), mode='constant', value=0)
    A = results - init_bone

    return new_J, A


def batch_quat_rotation(theta_quat):
    """
    Args:
        theta_quat: (N, 4)

    Returns:
        rotations: (N, num_joint, 3, 3)
    """

    x = theta_quat[:, 0]
    y = theta_quat[:, 1]
    z = theta_quat[:, 2]
    w = theta_quat[:, 3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    res_R = torch.stack([
        x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw),
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw),
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2
    ], dim=1).view(-1, 3, 3)

    return res_R


class SMPL(BaseSMPL):
    def __init__(self, model_path="./assets/checkpoints/pose3d/smpl_model.pkl", rotate=False):
        """
        pkl_path is the path to a SMPL model
        """
        super(SMPL, self).__init__()
        self.rotate = rotate

        # -- Load SMPL params --
        dd = load_pickle_file(model_path)

        # define faces
        self.faces = dd["f"].astype(np.uint64, copy=True)
        # self.register_buffer('faces', torch.from_numpy(undo_chumpy(dd['f']).astype(np.int32)).type(dtype=torch.int32))
        # self.faces = torch.from_numpy(dd['f'].astype(np.int32)).type(dtype=torch.int32)

        # Mean template vertices
        self.register_buffer('v_template', torch.FloatTensor(dd['v_template']))
        # Size of mesh [Number of vertices, 3], (6890, 3)
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis (shapedirs): (6980, 3, 10)
        # reshaped to (6980*3, 10), transposed to (10, 6980*3)
        self.register_buffer('shapedirs', torch.FloatTensor(np.reshape(
            dd['shapedirs'], [-1, self.num_betas]).T))

        # Regressor for joint locations given shape -> (24, 6890)
        # Transpose to shape (6890, 24)
        self.register_buffer('J_regressor', torch.FloatTensor(
            np.asarray(dd['J_regressor'].T.todense())))

        # Pose blend shape basis: (6890, 3, 207)
        num_pose_basis = dd['posedirs'].shape[-1]

        # Pose blend pose basis is reshaped to (6890*3, 207)
        # posedirs is transposed into (207, 6890*3)
        self.register_buffer('posedirs', torch.FloatTensor(np.reshape(
            dd['posedirs'], [-1, num_pose_basis]).T))

        # indices of parents for each joints
        self.parents = np.array(dd['kintree_table'][0].astype(np.int32))

        # LBS weights (6890, 24)
        self.register_buffer('weights', torch.FloatTensor(dd['weights']))

        # This returns 19 keypoints: 6890 x 19
        joint_regressor = torch.FloatTensor(
            np.asarray(dd['cocoplus_regressor'].T.todense()))

        self.register_buffer('joint_regressor', joint_regressor)

    def forward(self, beta, theta, offsets=0, links_ids=None, get_skin=False):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)
          offsets: N x 6890 x 3
          links_ids (None or list of np.ndarray): (from_verts_idx, to_verts_idx)
          get_skin: boolean, return skin or not

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6980 x 3
        """
        device = beta.device

        num_batch = beta.shape[0]

        # 1. Add shape blend shapes
        #       matmul  : (N, 10) x (10, 6890*3) = (N, 6890*3)
        #       reshape : (N, 6890*3) -> (N, 6890, 3)
        #       v_shaped: (N, 6890, 3)
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template + offsets

        # 2. Infer shape-dependent joint locations.
        # ----- J_regressor: (6890, 24)
        # ----- Jx (Jy, Jz): (N, 6890) x (6890, 24) = (N, 24)
        # --------------- J: (N, 24, 3)
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # 3. Add pose blend shapes
        # ------- theta    : (N, 72)   or (N, 96)
        # ------- reshape  : (N*24, 3) or (N * 24, 4)
        # ------- rodrigues: (N*24, 9) or quat rotations (N * 24, 9)
        # -- Rs = reshape  : (N, 24, 3, 3)
        if theta.shape[-1] == 72:  # Euler angles
            Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        elif theta.shape[-1] == 96:     # quat angles
            Rs = batch_quat_rotation(theta.view(-1, 4)).view(-1, 24, 3, 3)
        elif theta.shape[-1] == 144:    # 6D-rotation
            Rs = batch_rot6d_to_rotmat(theta.view(-1, 6)).view(-1, 24, 3, 3)
        else:   # rotation matrix, (24, 3, 3)
            Rs = theta
        # Ignore global rotation.
        #       Rs[:, 1:, :, :]: (N, 23, 3, 3)
        #           - np.eye(3): (N, 23, 3, 3)
        #          pose_feature: (N, 207)
        pose_feature = (Rs[:, 1:, :, :] - torch.eye(3).to(device)).view(-1, 207)

        # (N, 207) x (207, 6890*3) -> (N, 6890, 3)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped

        # 4. Get the global joint location
        # ------- Rs is (N, 24, 3, 3),         J is (N, 24, 3)
        # ------- J_transformed is (N, 24, 3), A is (N, 24, 4, 4)
        J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, device=device,
                                                             rotate_base=self.rotate)

        # 5. Do skinning:
        # ------- weights is (6890, 24)
        # ---------- tile is (N*6890, 24)
        # --- W = reshape is (N, 6890, 24)
        W = self.weights.repeat(num_batch, 1).view(num_batch, -1, 24)

        # ------ reshape A is (N, 24, 16)
        # --------- matmul is (N, 6890, 24) x (N, 24, 16) -> (N, 6890, 16)
        # -------- reshape is (N, 6890, 4, 4)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        # axis is 2, (N, 6890, 3) concatenate (N, 6890, 1) -> (N, 6890, 4)
        v_posed_homo = torch.cat(
            [v_posed, torch.ones(num_batch, v_posed.shape[1], 1, dtype=torch.float32).to(device)], dim=2)

        # -unsqueeze_ is (N, 6890, 4, 1)
        # --------- T is (N, 6890, 4, 4)
        # ---- matmul is (N, 6890, 4, 4) x (N, 6890, 4, 1) -> (N, 6890, 4, 1)
        v_posed_homo = v_posed_homo.unsqueeze(-1)
        v_homo = torch.matmul(T, v_posed_homo)

        # (N, 6890, 3)
        verts = v_homo[:, :, :3, 0]

        if links_ids is not None:
            verts = self.link(verts, links_ids)

        # Get cocoplus or lsp joints: (N, 6890) x (6890, 19)
        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints
