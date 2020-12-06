# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# Joint name to vertex mapping. SMPL/SMPL-H/SMPL-X vertices that correspond to
# MSCOCO, OpenPose (Body-25), Halpe joints
vertex_ids = {
    "smplh": {
        "nose":		    332,      # OpenPose-Body-25 (0),  Halpe-Body-26 (0)
        "reye":		    6260,     # OpenPose-Body-25 (15), Halpe-Body-26 (2)
        "leye":		    2800,     # OpenPose-Body-25 (16), Halpe-Body-26 (1)
        "rear":		    4071,     # OpenPose-Body-25 (17), Halpe-Body-26 (4)
        "lear":		    583,      # OpenPose-Body-25 (18), Halpe-Body-26 (3)
        "rthumb":		6191,     #
        "rindex":		5782,     #
        "rmiddle":		5905,     #
        "rring":		6016,     #
        "rpinky":		6133,     #
        "lthumb":		2746,     #
        "lindex":		2319,     #
        "lmiddle":		2445,     #
        "lring":		2556,     #
        "lpinky":		2673,     #
        "LBigToe":		3216,     # OpenPose-Body-25 (19), Halpe-Body-26 (20)
        "LSmallToe":	3226,     # OpenPose-Body-25 (20), Halpe-Body-26 (22)
        "LHeel":		3387,     # OpenPose-Body-25 (21), Halpe-Body-26 (24)
        "RBigToe":		6617,     # OpenPose-Body-25 (22), Halpe-Body-26 (21)
        "RSmallToe":    6624,     # OpenPose-Body-25 (23), Halpe-Body-26 (23)
        "RHeel":		6787      # OpenPose-Body-25 (24), Halpe-Body-26 (25)
    },
    "smplx": {
        "nose":		    9120,
        "reye":		    9929,
        "leye":		    9448,
        "rear":		    616,
        "lear":		    6,
        "rthumb":		8079,
        "rindex":		7669,
        "rmiddle":		7794,
        "rring":		7905,
        "rpinky":		8022,
        "lthumb":		5361,
        "lindex":		4933,
        "lmiddle":		5058,
        "lring":		5169,
        "lpinky":		5286,
        "LBigToe":		5770,
        "LSmallToe":    5780,
        "LHeel":		8846,
        "RBigToe":		8463,
        "RSmallToe": 	8474,
        "RHeel":  		8635
    }
}
