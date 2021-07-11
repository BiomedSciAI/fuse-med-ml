"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

from typing import Callable

import torch
import torch.nn as nn


def make_seq(foo: Callable, num: int, *args, **kwargs):
    """
    Makes a sequence of blocks
    """
    l = [foo(*args, **kwargs) for i in range(num)]
    return nn.Sequential(*l)


def make_final_seq(foo: Callable, num: int, *args, **kwargs):
    """
    Makes a sequence of blocks, but passes 'final_block'= True (to cut inside final block17)
    """
    l = [foo(*args, **kwargs) for i in range(num - 1)]
    kwargs['final_block'] = True
    l.append(foo(*args, **kwargs))
    return nn.Sequential(*l)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,
                                 momentum=0.01,  # changed from original 0.1
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out  # results in 96+64+96+64 channels


class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out  # results in 320 channels


class Mixed_6a(nn.Module):
    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out  # results in 384+384+320


class Block17(nn.Module):
    def __init__(self, scale=1.0, final_block=False, intra_block_cut_level=384):
        super(Block17, self).__init__()

        self.scale = scale
        self.final_block = final_block
        self.intra_block_cut_level = intra_block_cut_level

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        if self.final_block and self.intra_block_cut_level == 384:
            return out  # results in 384 channels

        out = self.conv2d(out)
        out = out * self.scale + x
        if self.final_block and self.intra_block_cut_level == 1088:
            return out  # results in 1088 channels
        out = self.relu(out)
        return out  # results in 1088 channels


class Mixed_7a(nn.Module):
    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out  # results in 384+288+320+1088


class Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out  # results in 2080 channels


class FuseBackboneInceptionResnetV2(nn.Module):
    def __init__(self,
                 logical_units_num: int = 14,
                 intra_block_cut_level: int = 384,
                 input_channels_num: int = 1) -> None:

        super().__init__()
        self.logical_units_num = logical_units_num
        self.intra_block_cut_level = intra_block_cut_level
        self.input_channels_num = input_channels_num

        # Modules
        self.conv2d_1a = BasicConv2d(input_channels_num, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.feature_depth = 96 + 64 + 96 + 64

        if self.logical_units_num >= 1:
            self.repeat = make_seq(Block35, min(self.logical_units_num - 1 + 1, 10), scale=0.17)
            self.feature_depth = 320

        if self.logical_units_num >= 11:
            self.mixed_6a = Mixed_6a()
            self.feature_depth = 384 + 384 + 320

        if self.logical_units_num >= 12:
            if 12 <= self.logical_units_num < 32:
                self.repeat_1 = make_final_seq(Block17, min(self.logical_units_num - 12 + 1, 20), scale=0.10,
                                               intra_block_cut_level=self.intra_block_cut_level)
                self.feature_depth = self.intra_block_cut_level

            else:
                self.repeat_1 = make_seq(Block17, min(self.logical_units_num - 12 + 1, 20), scale=0.10)
                self.feature_depth = 1088

        if self.logical_units_num >= 32:
            self.mixed_7a = Mixed_7a()
            self.feature_depth = 384 + 288 + 320 + 1088

        if self.logical_units_num >= 33:
            self.repeat_2 = make_seq(Block8, min(self.logical_units_num - 33 + 1, 9), scale=0.20)
            self.feature_depth = 2080

        if self.logical_units_num >= 42:
            self.block8 = Block8(noReLU=True)
            self.feature_depth = 2080

        if self.logical_units_num >= 43:
            self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
            self.feature_depth = 1536

    def features(self, input_tensor):
        x = self.conv2d_1a(input_tensor)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)

        if self.logical_units_num >= 1:
            x = self.repeat(x)

        if self.logical_units_num >= 11:
            x = self.mixed_6a(x)

        if self.logical_units_num >= 12:
            x = self.repeat_1(x)

        if self.logical_units_num >= 32:
            x = self.mixed_7a(x)

        if self.logical_units_num >= 33:
            x = self.repeat_2(x)

        if self.logical_units_num >= 42:
            x = self.block8(x)

        if self.logical_units_num >= 43:
            x = self.conv2d_7b(x)
        return x

    def forward(self, input_tensor):
        feature_map = self.features(input_tensor)  # typical features shape when cutting at level 14 is [batch_size, 384, H, W]
        return feature_map

    def get_feature_depth(self):
        return self.feature_depth
