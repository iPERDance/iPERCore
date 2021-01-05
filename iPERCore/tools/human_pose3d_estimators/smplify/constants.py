# original file comes from Copyright (c) 2019, University of Pennsylvania, Max Planck Institute for Intelligent Systems. All rights reserved.
# original file: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/pose.py
# this modified file Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

SMPL_JOINTS_NAMES = [
    # SMPL 24 Joints
    "Pelvis",      # 0  or Hip
    "LHip",        # 1
    "RHip",        # 2
    "Spine1",      # 3
    "LKnee",       # 4
    "RKnee",       # 5
    "Spine2",      # 6
    "LAnkle",      # 7
    "RAnkle",      # 8
    "Spine3",      # 9
    "LFoot",       # 10
    "RFoot",       # 11
    "Neck",        # 12
    "LCollar",     # 13
    "RCollar",     # 14
    "Head",        # 15
    "LShoulder",   # 16
    "RShoulder",   # 17
    "LElbow",      # 18
    "RElbow",      # 19
    "LWrist",      # 20
    "RWrist",      # 21
    "LHand",       # 22
    "RHand",       # 23

    # SMPL Vertices to Joints, 21
    "nose",         # 24
    "reye",         # 25
    "leye",         # 26
    "rear",         # 27
    "lear",         # 28
    "LBigToe",      # 29
    "LSmallToe",    # 30
    "LHeel",        # 31
    "RBigToe",      # 32
    "RSmallToe",    # 33
    "RHeel",        # 34
    "lthumb",       # 35
    "lindex",       # 36
    "lmiddle",      # 37
    "lring",        # 38
    "lpinky",       # 39
    "rthumb",       # 40
    "rindex",       # 41
    "rmiddle",      # 42
    "rring",        # 43
    "rpinky",       # 44

    # SMPL Vertices Regressing
    "Right Hip",    # 45, regressing Right Hip
    "Left Hip",     # 46, regressing Left Hip
    "Neck (LSP)",   # 47, regressing LSP Neck
    "Top of Head (LSP)",  # 48, regressing Top of LSP Head
    "Pelvis (MPII)",      # 49, regressing MPII Pelvis
    "Thorax (MPII)",      # 50, regressing MPII Thorax
    "Spine (H36M)",       # 51, regressing H36M Spine
    "Jaw (H36M)",         # 52, regressing H36M Jaw
    "Head (H36M)",        # 53, regressing H36M Head
]

"""
We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn"t provide annotations for a specific joint, we simply ignore it.
The joints used here are the following:
"""
JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    "OP Nose",
    "OP Neck",
    "OP RShoulder",
    "OP RElbow",
    "OP RWrist",
    "OP LShoulder",
    "OP LElbow",
    "OP LWrist",
    "OP MidHip",
    "OP RHip",
    "OP RKnee",
    "OP RAnkle",
    "OP LHip",
    "OP LKnee",
    "OP LAnkle",
    "OP REye",
    "OP LEye",
    "OP REar",
    "OP LEar",
    "OP LBigToe",
    "OP LSmallToe",
    "OP LHeel",
    "OP RBigToe",
    "OP RSmallToe",
    "OP RHeel",

    # 24 Ground Truth joints (superset of joints from different datasets)
    "Right Ankle",
    "Right Knee",
    "Right Hip",
    "Left Hip",
    "Left Knee",
    "Left Ankle",
    "Right Wrist",
    "Right Elbow",
    "Right Shoulder",
    "Left Shoulder",
    "Left Elbow",
    "Left Wrist",
    "Neck (LSP)",
    "Top of Head (LSP)",
    "Pelvis (MPII)",
    "Thorax (MPII)",
    "Spine (H36M)",
    "Jaw (H36M)",
    "Head (H36M)",
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",

    # 26 Halpe joints (in the order provided by Halpe)
    "Halpe Nose",  # 0
    "Halpe LEye",  # 1
    "Halpe REye",  # 2
    "Halpe LEar",  # 3
    "Halpe REar",  # 4
    "Halpe LShoulder",  # 5
    "Halpe RShoulder",  # 6
    "Halpe LElbow",  # 7
    "Halpe RElbow",  # 8
    "Halpe LWrist",  # 9
    "Halpe RWrist",  # 10
    "Halpe LHip",  # 11
    "Halpe RHip",  # 12
    "Halpe LKnee",  # 13
    "Halpe RKnee",  # 14
    "Halpe LAnkle",  # 15
    "Halpe RAnkle",  # 16
    "Halpe Head",  # 17, similar to `Top of Head` of LSP
    "Halpe Neck",  # 18
    "Halpe Hip",  # 19
    "Halpe LBigToe",  # 20
    "Halpe RBigToe",  # 21
    "Halpe LSmallToe",  # 22
    "Halpe RSmallToe",  # 23
    "Halpe LHeel",  # 24
    "Halpe RHeel",  # 25

    # 23 CocoWhole-Body joints (in the order provided by CocoWhole-Body)
    "CocoWholeBody Nose",        # 0
    "CocoWholeBody LEye",        # 1
    "CocoWholeBody REye",        # 2
    "CocoWholeBody LEar",        # 3
    "CocoWholeBody REar",        # 4
    "CocoWholeBody LShoulder",   # 5
    "CocoWholeBody RShoulder",   # 6
    "CocoWholeBody LElbow",      # 7
    "CocoWholeBody RElbow",      # 8
    "CocoWholeBody LWrist",      # 9
    "CocoWholeBody RWrist",      # 10
    "CocoWholeBody LHip",        # 11
    "CocoWholeBody RHip",        # 12
    "CocoWholeBody LKnee",       # 13
    "CocoWholeBody RKnee",       # 14
    "CocoWholeBody LAnkle",      # 15
    "CocoWholeBody RAnkle",      # 16
    "CocoWholeBody LBigToe",     # 17
    "CocoWholeBody LSmallToe",   # 18
    "CocoWholeBody LHeel",       # 19
    "CocoWholeBody RBigToe",     # 20
    "CocoWholeBody RSmallToe",   # 21
    "CocoWholeBody RHeel",       # 22
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL (45) joints
JOINT_MAP = {

    # OpenPose-Body-25
    "OP Nose": 24,              "OP Neck": 12,                  "OP RShoulder": 17,
    "OP RElbow": 19,            "OP RWrist": 21,                "OP LShoulder": 16,
    "OP LElbow": 18,            "OP LWrist": 20,                "OP MidHip": 0,
    "OP RHip": 2,               "OP RKnee": 5,                  "OP RAnkle": 8,
    "OP LHip": 1,               "OP LKnee": 4,                  "OP LAnkle": 7,
    "OP REye": 25,              "OP LEye": 26,                  "OP REar": 27,
    "OP LEar": 28,              "OP LBigToe": 29,               "OP LSmallToe": 30,
    "OP LHeel": 31,             "OP RBigToe": 32,               "OP RSmallToe": 33,
    "OP RHeel": 34,

    # CocoWhole-Body-23
    "CocoWholeBody Nose": 24,                                   "CocoWholeBody RShoulder": 17,
    "CocoWholeBody RElbow": 19, "CocoWholeBody RWrist": 21,     "CocoWholeBody LShoulder": 16,
    "CocoWholeBody LElbow": 18, "CocoWholeBody LWrist": 20,
    "CocoWholeBody RHip": 2,    "CocoWholeBody RKnee": 5,       "CocoWholeBody RAnkle": 8,
    "CocoWholeBody LHip": 1,    "CocoWholeBody LKnee": 4,       "CocoWholeBody LAnkle": 7,
    "CocoWholeBody REye": 25,   "CocoWholeBody LEye": 26,       "CocoWholeBody REar": 27,
    "CocoWholeBody LEar": 28,   "CocoWholeBody LBigToe": 29,    "CocoWholeBody LSmallToe": 30,
    "CocoWholeBody LHeel": 31,  "CocoWholeBody RBigToe": 32,    "CocoWholeBody RSmallToe": 33,
    "CocoWholeBody RHeel": 34,

    # Halpe-Body-26
    "Halpe Nose": 24,   "Halpe Neck": 12,    "Halpe RShoulder": 17,
    "Halpe RElbow": 19, "Halpe RWrist": 21,  "Halpe LShoulder": 16,
    "Halpe LElbow": 18, "Halpe LWrist": 20,  "Halpe Hip": 0,
    "Halpe RHip": 2,    "Halpe RKnee": 5,    "Halpe RAnkle": 8,
    "Halpe LHip": 1,    "Halpe LKnee": 4,    "Halpe LAnkle": 7,
    "Halpe REye": 25,   "Halpe LEye": 26,    "Halpe REar": 27,
    "Halpe LEar": 28,   "Halpe LBigToe": 29, "Halpe LSmallToe": 30,
    "Halpe LHeel": 31,  "Halpe RBigToe": 32, "Halpe RSmallToe": 33, "Halpe RHeel": 34,
    "Halpe Head": 48,

    # Other dataset
    "Right Ankle": 8, "Right Knee": 5, "Right Hip": 45,
    "Left Hip": 46, "Left Knee": 4, "Left Ankle": 7,
    "Right Wrist": 21, "Right Elbow": 19, "Right Shoulder": 17,
    "Left Shoulder": 16, "Left Elbow": 18, "Left Wrist": 20,
    "Neck (LSP)": 47, "Top of Head (LSP)": 48,
    "Pelvis (MPII)": 49, "Thorax (MPII)": 50,
    "Spine (H36M)": 51, "Jaw (H36M)": 52,
    "Head (H36M)": 53, "Nose": 24, "Left Eye": 26,
    "Right Eye": 25, "Left Ear": 28, "Right Ear": 27
}

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
# Indices to get the 14 LSP joints from the ground truth joints
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
J24_TO_J14 = J24_TO_J17[:14]

# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3 * i)
    SMPL_POSE_FLIP_PERM.append(3 * i + 1)
    SMPL_POSE_FLIP_PERM.append(3 * i + 2)
# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21] \
                + [25 + i for i in J24_FLIP_PERM]
