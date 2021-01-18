# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF


class ImageTransformer(object):
    """
    Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        """
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is matched to output_size.
                            If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        images = sample["images"]
        resized_images = []

        for image in images:
            height, width = image.shape[0:2]

            if height != self.output_size or width != self.output_size:
                image = cv2.resize(image, (self.output_size, self.output_size))

            image = image.astype(np.float32)
            image /= 255.0
            image = image * 2 - 1

            image = np.transpose(image, (2, 0, 1))

            resized_images.append(image)

        resized_images = np.stack(resized_images, axis=0)

        sample["images"] = resized_images
        return sample


class ImageNormalizeToTensor(object):
    """
    Rescale the image in a sample to a given size.
    """

    def __call__(self, image):
        # image = F.to_tensor(image)
        image = TF.to_tensor(image)
        image.mul_(2.0)
        image.sub_(1.0)
        return image


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        sample["images"] = torch.tensor(sample["images"]).float()
        sample["smpls"] = torch.tensor(sample["smpls"]).float()

        if "masks" in sample:
            sample["masks"] = torch.tensor(sample["masks"]).float()

        return sample

