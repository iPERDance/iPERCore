# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import numpy as np
from typing import List, Union, Tuple
from tqdm import tqdm

from iPERCore.tools.human_trackers import BaseHumanTracker


def get_largest_instance(human_instances: np.ndarray) -> Tuple[Union[np.ndarray, None], int]:
    """

    Get the largest area boxes from the instances.

    Args:
        human_instances (np.ndarray): (number of person, 5 = (x1, y1, x2, y2, cls_pred)). If number of person is 0,
            then it will return None.
    Returns:
        target_instance (np.ndarray or None): [1, 5 = (x1, y1, x2, y2, cls_pred)]. It returns None when the number of
            person is 0.
        target_ids (int): the target ids among number of people.
    """

    target_instance = None
    target_ids = 0

    num_person = human_instances.shape[0]

    if num_person > 0:
        max_area = 0

        for ids, instance in enumerate(human_instances):

            # x1, y1, x2, y2, conf, cls_conf, cls_pred = instance
            x1, y1, x2, y2 = instance[0:4]

            box_w = x2 - x1
            box_h = y2 - y1
            area = box_h * box_w

            if area > max_area:
                max_area = area
                target_instance = instance
                target_ids = ids

        target_instance = target_instance[np.newaxis]

    return target_instance, target_ids


class MaxBoxTracker(BaseHumanTracker):
    def __init__(self):
        pass

    def __call__(self, img, human_instances: np.ndarray) -> Tuple[Union[np.ndarray, None], int]:
        """

        Get the largest area boxes from the instances.

        Args:
            human_instances (np.ndarray): (number of person, 5 = (x1, y1, x2, y2, cls_pred)). If number of person is 0,
                then it will return None.
        Returns:
            target_instance (np.ndarray or None): [1, 5 = (x1, y1, x2, y2, cls_pred)]. It returns None when the number
                of person is 0.
            target_ids (int): the target ids among number of people.
        """

        target_instance, target_ids = get_largest_instance(human_instances)
        return target_instance, target_ids

    def run_tracker(self, human_detections: List[np.ndarray]) -> List[Union[np.ndarray, None]]:
        """

        Args:
            human_detections (List[np.ndarray]]): [(number of person, 5 = (x1, y1, x2, y2, cls_pred)),
                                                   (number of person, 5 = (x1, y1, x2, y2, cls_pred)),
                                                   ...,
                                                   (number of person, 5 = (x1, y1, x2, y2, cls_pred))]. If number of
                person is 0, then where is no person.

        Returns:
            --track_instances (list of np.ndarray or None): [np.array[[x1, y1, x2, y2, cls_pred]],
                                                             np.array[[x1, y1, x2, y2, cls_pred]],
                                                             ..., None,
                                                             np.array[[x1, y1, x2, y2, cls_pred]],
                                                             ..., None,
                                                             np.array[[x1, y1, x2, y2, cls_pred]]];
        """

        track_instances = []

        for human_instances in tqdm(human_detections):
            instance = get_largest_instance(human_instances)
            track_instances.append(instance)

        return track_instances
