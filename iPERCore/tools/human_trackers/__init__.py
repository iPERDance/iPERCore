# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import abc

VALID_TRACKERS = ["max_box"]


class BaseHumanTracker(object):

    @abc.abstractmethod
    def run_tracker(self, *args, **kwargs):
        pass


def build_tracker(name="max_box", *args, **kwargs):

    if name == "max_box":
        from .max_box_tracker import MaxBoxTracker

        tracker = MaxBoxTracker()

    else:
        raise ValueError(f"{name} is not valid, currently it only supports {VALID_TRACKERS}")

    return tracker
