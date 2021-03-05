# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm


pretrained_settings = {
    "resnet101": {
        "imagenet": {
            "input_space": "BGR",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.406, 0.456, 0.485],
            "std": [0.225, 0.224, 0.229],
            "num_classes": 1000
        }
    },
}

# colour map (21)
COLORS = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
    (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
    (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128)
]

DATASET_SETTINGS = {
    "atr": {
        "input_size": [512, 512],
        "num_classes": 18,
        "label": ["Background", "Hat", "Hair", "Sunglasses",
                  "Upper-clothes", "Skirt", "Pants", "Dress", "Belt",
                  "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg",
                  "Left-arm", "Right-arm", "Bag", "Scarf"],
        "background": [0, 16],  # remove "Bag"
        "body": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17],
        "skirt+dress": [5, 7]
    },
    "lip": {
        "input_size": [473, 473],
        "num_classes": 20,
        "label": ["Background", "Hat", "Hair", "Glove", "Sunglasses", "Upper-clothes", "Dress", "Coat",
                  "Socks", "Pants", "Jumpsuits", "Scarf", "Skirt", "Face", "Left-arm", "Right-arm",
                  "Left-leg", "Right-leg", "Left-shoe", "Right-shoe"],
        "background": [0],
        "body": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "skirt+dress": [6, 12]
    },
    "pascal-person": {
        "input_size": [512, 512],
        "num_classes": 7,
        "label": ["Background", "Head", "Torso", "Upper Arms", "Lower Arms", "Upper Legs", "Lower Legs"],
        "background": [0],
        "body": [1, 2, 3, 4, 5, 6],
        "skirt+dress": []
    }
}


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        # print(scale)
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def transform_logits(logits, center, scale, width, height, input_size):
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:, :, i],
            trans,
            (int(width), int(height)),  # (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))
        target_logits.append(target_logit)
    target_logits = np.stack(target_logits, axis=2)

    return target_logits


def find_largest_connected_mask(gray):
    """
        remove the isolated noisy, and return the largest connected mask.

    Args:
        gray (np.ndarray): size is (h, w), intensity is [0, 255], dtype is np.uint8

    Returns:
        mask (np.ndarray): the largest connected mask, size is (h, w), intensity is [0, 255], dtype is np.uint8.
    """

    mask = np.zeros_like(gray)

    # TODO, OpenCV 3.2-4.0
    # _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # TODO, OpenCV 4.0+
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    cv2.fillPoly(mask, [contours[max_idx]], 1)

    mask = mask * gray
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def get_trimap(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(mask, kernel, iterations=10)
    erode_mask = cv2.erode(mask, kernel, iterations=2)

    dilated_mask = dilated_mask.astype(np.float32)
    dilated_mask[dilated_mask != erode_mask] = 0.5

    return dilated_mask


def decode_parsing(pred_labels, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
      pred_labels (np.ndarray): result of inference after taking argmax.
      num_classes: number of classes to predict (including background).

    Returns:
      labels_color (np.ndarray): A batch with num_images RGB images of the same size as the input.
    """

    h, w = pred_labels.shape

    labels_color = np.zeros([h, w, 3], dtype=np.uint8)
    for i, c in enumerate(COLORS):
        c0 = labels_color[:, :, 0]
        c1 = labels_color[:, :, 1]
        c2 = labels_color[:, :, 2]

        flag = pred_labels == i

        c0[flag] = c[0]
        c1[flag] = c[1]
        c2[flag] = c[2]

    return labels_color


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class InPlaceABNSync(nn.Module):
    """
    Serve same as the InplaceABNSync.
    Reference: https://github.com/mapillary/inplace_abn
    """

    def __init__(self, num_features):
        super(InPlaceABNSync, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features=2048, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            InPlaceABNSync(out_features),
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode="bilinear", align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class EdgeModule(nn.Module):
    """
    Edge branch.
    """

    def __init__(self, in_fea=[256, 512, 1024], mid_fea=256, out_fea=2):
        super(EdgeModule, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode="bilinear", align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode="bilinear", align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode="bilinear", align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode="bilinear", align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)

        return edge, edge_fea


class DecoderModule(nn.Module):
    """
    Parsing Branch Decoder Module.

    """

    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(48)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode="bilinear", align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):

        self.inplanes = 128

        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)  # stride 16

        self.context_encoding = PSPModule()
        self.edge = EdgeModule()
        self.decoder = DecoderModule(num_classes)

        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=1))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Parsing Branch
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.context_encoding(x5)
        parsing_result, parsing_fea = self.decoder(x, x2)
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)
        fusion_result = self.fushion(x)
        return fusion_result


def initialize_pretrained_model(model, settings, pretrained="./models/resnet101-imagenet.pth"):
    model.input_space = settings["input_space"]
    model.input_size = settings["input_size"]
    model.input_range = settings["input_range"]
    model.mean = settings["mean"]
    model.std = settings["std"]

    if pretrained is not None:
        saved_state_dict = torch.load(pretrained)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split(".")
            if not i_parts[0] == "fc":
                new_params[".".join(i_parts[0:])] = saved_state_dict[i]
        model.load_state_dict(new_params)


def build_schp(num_classes=20, pretrained="./models/resnet101-imagenet.pth"):
    """

    Args:
        num_classes (int):
        pretrained (str or None):

    Returns:

    """
    global pretrained_settings

    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

    if pretrained is not None:
        settings = pretrained_settings["resnet101"]["imagenet"]
        initialize_pretrained_model(model, settings, pretrained)

    return model


class SCHPDataPreprocessor(object):
    def __init__(self, input_size=(512, 512)):
        self.input_size = input_size
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.input_size = np.asarray(input_size)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.406, 0.456, 0.485],  # BGR
                std=[0.225, 0.224, 0.229]  # BGR
            )
        ])

        self.root = None
        self.file_list = []

    def setup(self, root, file_list=None):
        assert not (root is None and file_list is None), f"{root} and {file_list} cannot both be None."

        self.root = root

        if file_list is None:
            self.file_list = sorted(os.listdir(root))
        else:
            self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2,), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):

        if self.root is None:
            img_path = self.file_list[index]
            img_name = os.path.split(img_path)[-1]
        else:
            img_name = self.file_list[index]
            img_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # # TODO: the original author"s version is BGR
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        img_np = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        # print(img_np.shape, img_np.max(), img_np.min())
        img_pt = self.transform(img_np)
        meta = {
            "name": img_name,
            "center": person_center,
            "height": h,
            "width": w,
            "scale": s,
            "rotation": r,
            # "orig_img": img_np
            "orig_img": img
        }

        return img_pt, meta


def remove_prefix_state_dict(state_dict, prefix="module"):
    """
    remove prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix) + 1:]] = state_dict[key].float()
    return new_state_dict


class SchpMattor(object):
    def __init__(self,
                 restore_weight="./assets/checkpoints/mattors/exp-schp-lip.pth",
                 device=torch.device("cuda:0")):

        self.device = device
        self.restore_weight = restore_weight
        self.dataset = "lip"
        self.dataset_info = DATASET_SETTINGS[self.dataset]

        num_classes = self.dataset_info["num_classes"]
        input_size = self.dataset_info["input_size"]
        label = self.dataset_info["label"]

        # build model
        model = build_schp(num_classes=num_classes, pretrained=None).to(device)
        model = nn.DataParallel(model)
        state_dict = torch.load(restore_weight)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model.to(device)

        # build data preprocessor
        self.data_preprocessor = SCHPDataPreprocessor(input_size)

    def run(self, src_dir, out_dir, src_file_list=None, target="body", save_visual=True):
        """

        Args:
            src_dir (str or None):
            out_dir (str or None):
            src_file_list (list of str or None):
            target (str): "body" or "skirt+dress";
            save_visual (bool):

        Returns:
            flag (bool): if it detects mask, then it returns True, otherwise it will return False.
            mask_outs (list of str or list of np.ndarray):
            alpha_outs (list of str or list of np.ndarray):
        """

        mask_outs = []
        alpha_outs = []

        self.data_preprocessor.setup(src_dir, src_file_list)

        if out_dir is not None:
            os.makedirs(out_dir)

        background = self.dataset_info["background"]
        num_classes = self.dataset_info["num_classes"]
        target_classes = self.dataset_info[target]

        valid = np.zeros((num_classes,), dtype=np.uint8)
        valid[target_classes] = 1

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.data_preprocessor)):
                image, meta = batch
                image = image[None].to(self.device)

                img_name = meta["name"]
                c = meta["center"]
                s = meta["scale"]
                w = meta["width"]
                h = meta["height"]

                prefix_name = img_name.split(".")[0]
                # print(w, h, image.shape, meta["orig_img"].shape)

                output = self.model(image)
                upsample = torch.nn.Upsample(size=(h, w), mode="bilinear", align_corners=True)
                upsample_output = upsample(output)
                upsample_output = upsample_output.squeeze()
                upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

                logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=(h, w))
                raw_parse = np.argmax(logits_result, axis=2)
                front_mask = valid[raw_parse].copy()

                # if there is no more than 100 pixels, then we assume there is no skirt or dress.
                # print(f"number of front mask {np.sum(front_mask)}")
                if target == "skirt+dress" and np.sum(front_mask) < 100:
                    return False, mask_outs, alpha_outs

                lcc_mask = find_largest_connected_mask(front_mask)

                if out_dir is None:
                    mask_outs.append(lcc_mask)
                else:
                    mask_path = os.path.join(out_dir, prefix_name + "_mask.png")
                    cv2.imwrite(mask_path, (lcc_mask * 255).astype(np.uint8))
                    mask_outs.append(mask_path)

                if save_visual:
                    raw_out_img = decode_parsing(raw_parse)
                    cv2.imwrite(os.path.join(out_dir, prefix_name + "_raw_parse.png"), raw_out_img)
                    cv2.imwrite(os.path.join(out_dir, prefix_name + "_lcc_mask.png"), (lcc_mask * 255).astype(np.uint8))

        return True, mask_outs, alpha_outs


if __name__ == "__main__":
    device = torch.device("cuda:0")
    human_parser = SchpMattor(
        restore_weight="./assets/pretrains/exp-schp-pascal-person-part.pth",
        device=device
    )

    # human_parser = HumanParser(
    #     dataset="atr",
    #     restore_weight="./assets/pretrains/exp-schp-atr.pth",
    #     device=device
    # )

    root_dir = "/p300/tpami/neuralAvatar/experiments/wukong.jpg/processed"
    # root_dir = "/p300/tpami/neuralAvatar/experiments/liuwen/processed"
    src_dir = os.path.join(root_dir, "images")
    out_dir = os.path.join(root_dir, "parse")

    human_parser.run(src_dir, out_dir, save_visual=True)
