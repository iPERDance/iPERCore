# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

from collections import OrderedDict

import torch
import torch.nn as nn


def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


def body25_make_layers(block, act_types):
    """

    Args:
        block:
        act_types (dict):

    Returns:
        module (nn.Sequential): the module layer.
    """

    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        elif 'prelu' in layer_name:
            layers.append((layer_name, nn.PReLU(num_parameters=v[0])))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name in act_types:
                act = nn.ReLU(inplace=True)
                layers.append((act_types[layer_name] + '_' + layer_name, act))

    return nn.Sequential(OrderedDict(layers))


class MConvBlock(nn.Module):
    def __init__(self, conv_ids, stage_ids, l_name, in_channel, out_channel,
                 kernel_size=3, stride=1, padding=1,
                 is_single=False, has_relu=True):
        """

        Args:
            conv_ids:
            stage_ids:
            l_name:
            in_channel:
            out_channel:
            is_single:
            has_relu:
        """
        super().__init__()

        self.is_single = is_single

        if self.is_single:
            name_template = "M{layer}{conv_ids}_stage{stage_ids}_L{l_name}"

            m_split = list()
            m_split.append(
                (name_template.format(layer="conv", conv_ids=conv_ids, stage_ids=stage_ids, l_name=l_name),
                 nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                           kernel_size=kernel_size, stride=stride, padding=padding))
            )
            if has_relu:
                m_split.append(
                    (name_template.format(layer="prelu", conv_ids=conv_ids, stage_ids=stage_ids, l_name=l_name),
                     nn.PReLU(num_parameters=out_channel))
                )
            self.split0 = nn.Sequential(OrderedDict(m_split))
        else:
            name_template = "M{layer}{conv_ids}_stage{stage_ids}_L{l_name}_{col_num}"
            conv_0 = [
                (name_template.format(layer="conv", conv_ids=conv_ids, stage_ids=stage_ids,
                                      l_name=l_name, col_num=0),
                 nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                           kernel_size=kernel_size, stride=stride, padding=padding)),

                (name_template.format(layer="prelu", conv_ids=conv_ids, stage_ids=stage_ids,
                                      l_name=l_name, col_num=0),
                 nn.PReLU(num_parameters=out_channel))
            ]
            self.split0 = nn.Sequential(OrderedDict(conv_0))

            conv_1 = [
                (name_template.format(layer="conv", conv_ids=conv_ids, stage_ids=stage_ids,
                                      l_name=l_name, col_num=1),
                 nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                           kernel_size=kernel_size, stride=stride, padding=padding)),

                (name_template.format(layer="prelu", conv_ids=conv_ids, stage_ids=stage_ids,
                                      l_name=l_name, col_num=1),
                 nn.PReLU(num_parameters=out_channel))
            ]
            self.split1 = nn.Sequential(OrderedDict(conv_1))

            conv_2 = [
                (name_template.format(layer="conv", conv_ids=conv_ids, stage_ids=stage_ids,
                                      l_name=l_name, col_num=2),
                 nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                           kernel_size=kernel_size, stride=stride, padding=padding)),

                (name_template.format(layer="prelu", conv_ids=conv_ids, stage_ids=stage_ids,
                                      l_name=l_name, col_num=2),
                 nn.PReLU(num_parameters=out_channel))
            ]
            self.split2 = nn.Sequential(OrderedDict(conv_2))

    def forward(self, x):
        conv0 = self.split0(x)

        if not self.is_single:
            conv1 = self.split1(conv0)
            conv2 = self.split2(conv1)
            out = torch.cat([conv0, conv1, conv2], dim=1)
        else:
            out = conv0

        return out


class StackMConvBlock(nn.Module):
    def __init__(self, stage, stage_params):
        """

        Args:
            stage (int):
            stage_params (dict):
                stage_params = {
                    # layer_num: [in_channel, out_channel, kernel_size, stride, padding, is_single, has_relu, l_name]
                    1: [128, 96, 3, 1, 1, False, True, 2],
                    2: [96 * 3, 96, 3, 1, 1, False, True, 2],
                    3: [96 * 3, 96, 3, 1, 1, False, True, 2],
                    4: [96 * 3, 96, 3, 1, 1, False, True, 2],
                    5: [96 * 3, 96, 3, 1, 1, False, True, 2],
                    6: [96 * 3, 256, 1, 1, 0, True, True, 2],
                    7: [256, 52, 1, 1, 0, True, False, 2],
                }
        """
        super().__init__()

        # the first five mconv block
        blocks = []
        for i in range(1, 8):
            in_channel, out_channel, kernel_size, stride, padding, is_single, has_relu, l_name = stage_params[i]
            mblock = MConvBlock(conv_ids=i, stage_ids=stage, l_name=l_name, in_channel=in_channel,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                out_channel=out_channel, is_single=is_single, has_relu=has_relu)
            blocks.append(mblock)

        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        return self.main(x)


class OpenPoseBody25Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(OpenPoseBody25Model, self).__init__()

        # model 0
        self.model0 = self.build_model0()

        # M-stage0_L2
        stage_0_L2_params = {
            # layer_num: [in_channel, out_channel, kernel_size, stride, padding, is_single, has_relu]
            1: [128, 96, 3, 1, 1, False, True, 2],
            2: [96 * 3, 96, 3, 1, 1, False, True, 2],
            3: [96 * 3, 96, 3, 1, 1, False, True, 2],
            4: [96 * 3, 96, 3, 1, 1, False, True, 2],
            5: [96 * 3, 96, 3, 1, 1, False, True, 2],
            6: [96 * 3, 256, 1, 1, 0, True, True, 2],
            7: [256, 52, 1, 1, 0, True, False, 2],
        }
        self.block02 = StackMConvBlock(stage=0, stage_params=stage_0_L2_params)

        # M-stage1_L2
        stage_1_L2_params = {
            # layer_num: [in_channel, out_channel, kernel_size, stride, padding, is_single, has_relu]
            1: [180, 128, 3, 1, 1, False, True, 2],
            2: [128 * 3, 128, 3, 1, 1, False, True, 2],
            3: [128 * 3, 128, 3, 1, 1, False, True, 2],
            4: [128 * 3, 128, 3, 1, 1, False, True, 2],
            5: [128 * 3, 128, 3, 1, 1, False, True, 2],
            6: [128 * 3, 512, 1, 1, 0, True, True, 2],
            7: [512, 52, 1, 1, 0, True, False, 2],
        }
        self.block12 = StackMConvBlock(stage=1, stage_params=stage_1_L2_params)

        # M-stage2_L2
        stage_2_L2_params = stage_1_L2_params
        self.block22 = StackMConvBlock(stage=2, stage_params=stage_2_L2_params)

        # M-stage3_L2
        stage_3_L2_params = stage_1_L2_params
        self.block32 = StackMConvBlock(stage=3, stage_params=stage_3_L2_params)

        # M-stage0_L1
        stage_0_L1_params = {
            # layer_num: [in_channel, out_channel, kernel_size, stride, padding, is_single, has_relu]
            1: [180, 96, 3, 1, 1, False, True, 1],
            2: [96 * 3, 96, 3, 1, 1, False, True, 1],
            3: [96 * 3, 96, 3, 1, 1, False, True, 1],
            4: [96 * 3, 96, 3, 1, 1, False, True, 1],
            5: [96 * 3, 96, 3, 1, 1, False, True, 1],
            6: [96 * 3, 256, 1, 1, 0, True, True, 1],
            7: [256, 26, 1, 1, 0, True, False, 1],
        }
        self.block01 = StackMConvBlock(stage=0, stage_params=stage_0_L1_params)

        # M-stage1_L1
        stage_1_L1_params = {
            # layer_num: [in_channel, out_channel, kernel_size, stride, padding, is_single, has_relu]
            1: [206, 128, 3, 1, 1, False, True, 1],
            2: [128 * 3, 128, 3, 1, 1, False, True, 1],
            3: [128 * 3, 128, 3, 1, 1, False, True, 1],
            4: [128 * 3, 128, 3, 1, 1, False, True, 1],
            5: [128 * 3, 128, 3, 1, 1, False, True, 1],
            6: [128 * 3, 512, 1, 1, 0, True, True, 1],
            7: [512, 26, 1, 1, 0, True, False, 1],
        }
        self.block11 = StackMConvBlock(stage=1, stage_params=stage_1_L1_params)

    def build_model0(self):
        # Stage 0
        act_types = {
            'conv1_1': 'relu', 'conv1_2': 'relu', 'conv2_1': 'relu', 'conv2_2': 'relu',
            'conv3_1': 'relu', 'conv3_2': 'relu', 'conv3_3': 'relu', 'conv3_4': 'relu', 'conv4_1': 'relu'
        }

        block0 = OrderedDict([
            ('conv1_1', [3, 64, 3, 1, 1]),
            ('conv1_2', [64, 64, 3, 1, 1]),
            ('pool1_stage1', [2, 2, 0]),
            ('conv2_1', [64, 128, 3, 1, 1]),
            ('conv2_2', [128, 128, 3, 1, 1]),
            ('pool2_stage1', [2, 2, 0]),
            ('conv3_1', [128, 256, 3, 1, 1]),
            ('conv3_2', [256, 256, 3, 1, 1]),
            ('conv3_3', [256, 256, 3, 1, 1]),
            ('conv3_4', [256, 256, 3, 1, 1]),
            ('pool3_stage1', [2, 2, 0]),
            ('conv4_1', [256, 512, 3, 1, 1]),
            ('conv4_2', [512, 512, 3, 1, 1]),
            ('prelu4_2', [512]),
            ('conv4_3_CPM', [512, 256, 3, 1, 1]),
            ('prelu4_3_CPM', [256]),
            ('conv4_4_CPM', [256, 128, 3, 1, 1]),
            ('prelu4_4_CPM', [128])
        ])

        model0 = body25_make_layers(block0, act_types)
        return model0

    def forward(self, x):
        out1 = self.model0(x)

        # M-stage0-L2
        m_stage_0_L2 = self.block02(out1)

        # M-stage1-L2
        concat_stage1_L2 = torch.cat([out1, m_stage_0_L2], dim=1)
        m_stage_1_L2 = self.block12(concat_stage1_L2)

        # M-stage2-L2
        concat_stage2_L2 = torch.cat([out1, m_stage_1_L2], dim=1)
        m_stage_2_L2 = self.block22(concat_stage2_L2)

        # M-stage3-L2
        concat_stage3_L2 = torch.cat([out1, m_stage_2_L2], dim=1)
        Mconv7_stage3_L2 = self.block32(concat_stage3_L2)

        # M-stage0-L1
        concat_stage0_L1 = torch.cat([out1, Mconv7_stage3_L2], dim=1)
        Mconv7_stage0_L1 = self.block01(concat_stage0_L1)

        # M-stage1-L1
        # concat_stage1_L1 = torch.cat([out1, m_stage_3_L2, m_stage_0_L1], dim=1)
        concat_stage1_L1 = torch.cat([out1, Mconv7_stage0_L1, Mconv7_stage3_L2], dim=1)

        Mconv7_stage1_L1 = self.block11(concat_stage1_L1)

        # outputs = {
        #     "heatMat": Mconv7_stage1_L1,
        #     "pafMat": Mconv7_stage3_L2
        # }

        return Mconv7_stage1_L1, Mconv7_stage3_L2


class OpenPoseBody18Model(nn.Module):
    def __init__(self):
        super(OpenPoseBody18Model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict([
            ('conv1_1', [3, 64, 3, 1, 1]),
            ('conv1_2', [64, 64, 3, 1, 1]),
            ('pool1_stage1', [2, 2, 0]),
            ('conv2_1', [64, 128, 3, 1, 1]),
            ('conv2_2', [128, 128, 3, 1, 1]),
            ('pool2_stage1', [2, 2, 0]),
            ('conv3_1', [128, 256, 3, 1, 1]),
            ('conv3_2', [256, 256, 3, 1, 1]),
            ('conv3_3', [256, 256, 3, 1, 1]),
            ('conv3_4', [256, 256, 3, 1, 1]),
            ('pool3_stage1', [2, 2, 0]),
            ('conv4_1', [256, 512, 3, 1, 1]),
            ('conv4_2', [512, 512, 3, 1, 1]),
            ('conv4_3_CPM', [512, 256, 3, 1, 1]),
            ('conv4_4_CPM', [256, 128, 3, 1, 1])
        ])

        # Stage 1
        block1_1 = OrderedDict([
            ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
            ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
        ])

        block1_2 = OrderedDict([
            ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
            ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
        ])
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict([
                ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
            ])

            blocks['block%d_2' % i] = OrderedDict([
                ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
            ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']

    def forward(self, x):

        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2


class OpenPoseHandModel(nn.Module):
    def __init__(self):
        super(OpenPoseHandModel, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        # stage 1
        block1_0 = OrderedDict([
            ('conv1_1', [3, 64, 3, 1, 1]),
            ('conv1_2', [64, 64, 3, 1, 1]),
            ('pool1_stage1', [2, 2, 0]),
            ('conv2_1', [64, 128, 3, 1, 1]),
            ('conv2_2', [128, 128, 3, 1, 1]),
            ('pool2_stage1', [2, 2, 0]),
            ('conv3_1', [128, 256, 3, 1, 1]),
            ('conv3_2', [256, 256, 3, 1, 1]),
            ('conv3_3', [256, 256, 3, 1, 1]),
            ('conv3_4', [256, 256, 3, 1, 1]),
            ('pool3_stage1', [2, 2, 0]),
            ('conv4_1', [256, 512, 3, 1, 1]),
            ('conv4_2', [512, 512, 3, 1, 1]),
            ('conv4_3', [512, 512, 3, 1, 1]),
            ('conv4_4', [512, 512, 3, 1, 1]),
            ('conv5_1', [512, 512, 3, 1, 1]),
            ('conv5_2', [512, 512, 3, 1, 1]),
            ('conv5_3_CPM', [512, 128, 3, 1, 1])
        ])

        block1_1 = OrderedDict([
            ('conv6_1_CPM', [128, 512, 1, 1, 0]),
            ('conv6_2_CPM', [512, 22, 1, 1, 0])
        ])

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict([
                ('Mconv1_stage%d' % i, [150, 128, 7, 1, 3]),
                ('Mconv2_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d' % i, [128, 22, 1, 1, 0])
            ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6


def build_openpose_model(name="OpenPose-Body-25"):

    if name == "OpenPose-Body-25":
        model = OpenPoseBody25Model()

    elif name == "OpenPose-Body-18":
        model = OpenPoseBody18Model()

    elif name == "OpenPose-Hand":
        model = OpenPoseHandModel()

    else:
        raise ValueError(f"{name} is not valid.")

    return model
