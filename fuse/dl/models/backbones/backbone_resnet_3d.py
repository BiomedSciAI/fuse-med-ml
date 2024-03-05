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

import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Callable, List, Sequence, Type


class Conv3DSimple(nn.Conv3d):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        midplanes: Optional[int] = None,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, stride, stride


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super().__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes), nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: Tuple[int, int, int] = (3, 7, 7),
        stride: Tuple[int, int, int] = (1, 2, 2),
    ):
        padding = tuple([x // 2 for x in kernel_size])
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


class BackboneResnet3D(nn.Module):
    """
    A slightly more configureable ResNet3D implementation.
    Default values are identical to the pytorch implementation.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        pool: bool = False,
        layers: List[int] = [2, 2, 2, 2],
        first_channel_dim: int = 64,
        first_stride: int = 1,
        stem_kernel_size: Sequence[int] = (3, 7, 7),
        stem_stride: Sequence[int] = (1, 2, 2),
        pretrained: bool = False,
        name: str = "r3d_18",
    ) -> None:
        """
        :param in_channels: number of channels in the image
        :param pool: whether to use global average pooling to reduce the spacial dimensions after the convolutional layers

        :param layers: number of resnet blocks in every layer
        :param first_channel_dim: number of channels in the first layer. Every layer increases the channels by x2.
        :param first_stride: the stride in the first layer. using 2 will downsample the image by x2 in every spatial axis.

        :param stem_kernel_size: kernel size for the stem (first convolution)
        :param stem_stride: stride of the stem in every axis.

        :param pretrained: if True loads the pretrained video resnet model.
        :param name: name of the model to load, relevant only if pretrained=True
        """
        super().__init__()
        self.inplanes = first_channel_dim
        self.stem = BasicStem(
            3, first_channel_dim, kernel_size=stem_kernel_size, stride=stem_stride
        )

        self.layer1 = self._make_layer(
            BasicBlock, Conv3DSimple, first_channel_dim, layers[0], stride=first_stride
        )
        self.layer2 = self._make_layer(
            BasicBlock, Conv3DSimple, first_channel_dim * 2, layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            BasicBlock, Conv3DSimple, first_channel_dim * 4, layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            BasicBlock, Conv3DSimple, first_channel_dim * 8, layers[3], stride=2
        )
        self.out_dim = first_channel_dim * 8
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self._pool = pool
        self._pretrained = pretrained

        if pretrained:
            print("Warning: not supported by new torchvision version")
            from torch.hub import load_state_dict_from_url
            from torchvision.models.video.resnet import model_urls

            state_dict = load_state_dict_from_url(model_urls[name])
            del state_dict["fc.weight"]  # as a backbone the fc in not necessary
            del state_dict["fc.bias"]
            self.load_state_dict(state_dict)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        if in_channels == 1:
            self.stem[0].in_channels = 1
            self.stem[0].weight = nn.Parameter(
                self.stem[0].weight.sum(dim=1, keepdim=True)
            )
        elif in_channels != 3:
            self.stem = BasicStem(
                in_channels,
                first_channel_dim,
                kernel_size=stem_kernel_size,
                stride=stem_stride,
            )

    def _make_layer(
        self,
        block: Type[BasicBlock],
        conv_builder: Type[Conv3DSimple],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=ds_stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self._pool:
            x = self.avgpool(x)
            x = x.flatten(1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass. 3D global classification given a volume
        :param x: Input volume. shape: [batch_size, channels, z, y, x]
        :return: if pool is True:  feature vector per sample. shape: [batch_size, first_channel_dim * 8].
                 if pool is False: a 5D tensor of shape [batch_size, first_channel_dim * 8, z, y, x]
                                   where z,y,x are x8 or x16 smaller than the original dim (depends on the strides used)
        """
        return self.features(x)
