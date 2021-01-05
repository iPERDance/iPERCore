# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import abc
from enum import Enum, unique


@unique
class ACTIONS(Enum):
    SPLIT = 0
    SKIN = 1
    DETAIL = 2
    SMPL = 3


class BasePose3dRunner(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def run_with_smplify(self, *args, **kwargs):
        pass


def build_pose3d_estimator(name: str, *args, **kwargs):
    """

    Args:
        name (str):
        *args:
        **kwargs:

    Returns:
        estimator (BaseRunner):
    """

    if name == "spin":
        from .spin import SPINRunner
        estimator = SPINRunner(*args, **kwargs)

    else:
        raise ValueError(name)

    return estimator


class BasePose3dRefiner(metaclass=abc.ABCMeta):

    def __init__(self):
        self.formater = None

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass

    @property
    def keypoint_formater(self):
        return self.formater


def build_pose3d_refiner(name: str, *args, **kwargs):
    """

    Args:
        name:
        *args:
        **kwargs:

    Returns:
        estimator (BaseRunner):
    """
    if name == "smplify":
        from .smplify import SMPLifyRunner
        estimator = SMPLifyRunner(*args, **kwargs)

    else:
        raise ValueError(name)

    return estimator
