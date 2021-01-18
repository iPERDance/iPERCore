# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

VALID_NAMES = ["BODY_25", "MobileNet"]


def build_openpose_model(name="mobilenet", *args, **kwargs):

    if name == "BODY_25":
        from .openposenet import OpenPoseBody25Model
        network = OpenPoseBody25Model(*args, **kwargs)

    elif name == "MobileNet":
        from .mobilenet import PoseEstimationWithMobileNet
        network = PoseEstimationWithMobileNet(*args, **kwargs)

    else:
        raise ValueError(f"Invalid name {name}, and currently, it only support {VALID_NAMES}.")

    return network
