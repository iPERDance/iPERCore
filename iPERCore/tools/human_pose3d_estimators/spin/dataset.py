import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from iPERCore.tools.utils.geometry.keypoints import KeypointFormater

__all__ = ["preprocess", "InferenceDatasetWithKeypoints", "InferenceDataset"]


def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * np.float(scale_factor))).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def preprocess(img, boxes):
    """
    Args:
        img (np.ndarray): (height, width, 3) is in the range of [0, 255], BGR channel.
        boxes (np.ndarray, List, or Tuple) : (4,) = [x0, y0, x1, y1]

    Returns:
        proc_img (np.ndarray): the cropped image, (3, 224, 224) is in the range of [0, 1], RGB channel;
        proc_info (dict): the information of processed image, and it contains:
            --im_shape (tuple): the original shape (height, width);
            --center (np.ndarray): (cx, cy), the center of the processed image;
            --scale (float): the scale to resize the original image to the current size;
            --start_pt (np.ndarray): the start points, and it is used to convert camera in the cropped image
                coordinates to the original image coordinates, see the function `cam_init2orig`.
    """
    image_size = 224

    orig_h, orig_w = img.shape[0:2]
    x0, y0, x1, y1 = boxes

    size = max(x1 - x0, y1 - y0)
    scale = 200 / size

    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    image_scaled, scale_factors = resize_img(img, scale)

    center_scaled = np.round(center * scale_factors).astype(np.int)

    # Make sure there is enough space to crop image_size x image_size.
    image_padded = np.pad(
        array=image_scaled,
        pad_width=((image_size,), (image_size,), (0,)),
        mode='constant'
    )
    padded_h, padded_w = image_padded.shape[0:2]
    center_scaled += image_size

    # Crop image_size x image_size around the center.
    margin = image_size // 2
    start_pt = (center_scaled - margin).astype(int)
    end_pt = (center_scaled + margin).astype(int)
    end_pt[0] = min(end_pt[0], padded_w)
    end_pt[1] = min(end_pt[1], padded_h)

    proc_img = image_padded[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
    proc_img = proc_img.astype(np.float32) / 255
    proc_img = np.transpose(proc_img, (2, 0, 1))
    # height, width = image_scaled.shape[:2]

    center_scaled -= start_pt

    proc_info = {
        'im_shape': (orig_h, orig_w),
        'center': center_scaled,
        'scale': scale,
        'start_pt': start_pt,
    }

    return proc_img, proc_info


class InferenceDataset(Dataset):
    def __init__(self, image_paths, boxes, image_size=224):
        super(InferenceDataset, self).__init__()
        self.image_paths = image_paths
        self.boxes = boxes
        self.image_size = image_size
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        """

        Args:
            item (int):

        Returns:
            sample (dict): the sample information, it contains,
              --image (torch.Tensor): (3,  224, 224) is the cropped image range of [0, 1] and normalized
                    by MEAN and STD, RGB channel;
              --orig_image (torch.Tensor): (3, height, width) is the in rage of [0, 1], RGB channel;
              --im_shape (torch.Tensor): (height, width)
              --center (torch.Tensor): (2,);
              --start_pt (torch.Tensor): (2,);
              --scale (torch.Tensor): (1,);
              --img_path (str): the image path.
        """

        # 1. load image
        img_path = self.image_paths[item]
        image = cv2.imread(img_path)

        # 2. load boxes
        boxes = np.copy(self.boxes[item])

        proc_img, sample = preprocess(image, boxes)
        proc_img = torch.from_numpy(proc_img)
        proc_img = self.normalize(proc_img)
        sample["image"] = proc_img
        sample["orig_image"] = image

        sample["im_shape"] = torch.tensor(sample["im_shape"]).float()
        sample["img_path"] = img_path
        sample["img_id"] = item

        return sample


class InferenceDatasetWithKeypoints(Dataset):
    def __init__(self, image_paths, boxes, keypoints, keypoint_formater, temporal=False, image_size=224):

        """

        Args:
            image_paths:
            boxes:
            keypoints:
            keypoint_formater (KeypointFormater):
            temporal:
            image_size:
        """
        super().__init__()
        self.image_paths = image_paths
        self.boxes = boxes
        self.image_size = image_size
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.temporal = temporal

        self.keypoint_formater = keypoint_formater

        keypoints_info = self.keypoint_formater.stack_keypoints(keypoints)

        # # if there are more than 10 frames, then we will use temporal smooth of smpl.
        # if temporal:
        #     keypoints_info = self.keypoint_formater.temporal_smooth_keypoints(keypoints_info)

        self.keypoints_info = keypoints_info

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        """

        Args:
            item (int):

        Returns:
            sample (dict): the sample information, it contains,
              --image (torch.Tensor): (3,  224, 224) is the cropped image range of [0, 1] and normalized
                    by MEAN and STD, RGB channel;
              --orig_image (torch.Tensor): (3, height, width) is the in rage of [0, 1], RGB channel;
              --im_shape (torch.Tensor): (height, width)
              --keypoints (dict): the keypoints information, it contains,
                --pose_keypoints_2d (torch.Tensor): (25, 3)
                --face_keypoints_2d (torch.Tensor):
                --hand_left_keypoints_2d (torch.Tensor):
                --hand_right_keypoints_2d (torch.Tensor):
              --center (torch.Tensor): (2,);
              --start_pt (torch.Tensor): (2,);
              --scale (torch.Tensor): (1,);
              --img_path (str): the image path.
        """

        # 1. load image
        img_path = self.image_paths[item]
        image = cv2.imread(img_path)

        # 2. load boxes
        boxes = np.copy(self.boxes[item])

        proc_img, sample = preprocess(image, boxes)
        proc_img = torch.from_numpy(proc_img)
        proc_img = self.normalize(proc_img)
        sample["image"] = proc_img
        sample["orig_image"] = image

        # 3. load keypoints
        # sample["keypoints"] = self.format_keypoints(deepcopy(self.keypoints[item]), sample["im_shape"])
        sample["keypoints"] = self.keypoint_formater.format_stacked_keypoints(
            item, self.keypoints_info,
            sample["im_shape"]
        )

        sample["im_shape"] = torch.tensor(sample["im_shape"]).float()
        sample["img_path"] = img_path
        sample["img_id"] = item

        return sample
