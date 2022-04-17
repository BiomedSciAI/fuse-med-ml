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

from typing import Tuple, Any

import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torchvision.models.video.resnet import VideoResNet, BasicBlock, Conv3DSimple, BasicStem, model_urls


class FuseBackboneResnet3D(VideoResNet):
    """
    3D model classifier (ResNet architecture"
    """

    def __init__(self, pretrained: bool = False, in_channels: int = 2, name: str = "r3d_18") -> None:
        """
        Create 3D ResNet model
        :param pretrained: Use pretrained weights
        :param in_channels: number of input channels
        :param name: model name. currently only 'r3d_18' is supported
        """
        # init parameters per required backbone
        init_parameters = {
            'r3d_18': {'block': BasicBlock,
                       'conv_makers': [Conv3DSimple] * 4,
                       'layers': [2, 2, 2, 2],
                       'stem': BasicStem},
        }[name]
        # init original model
        super().__init__(**init_parameters)

        # load pretrained parameters if required
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[name])
            self.load_state_dict(state_dict)

        # save input parameters
        self.pretrained = pretrained
        self.in_channels = in_channels
        # override the first convolution layer to support any number of input channels
        self.stem = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

    def features(self, x: Tensor) -> Any:
        """
        Extract spatial features - given a 3D tensor
        :param x: Input tensor - shape: [batch_size, channels, z, y, x]
        :return: spatial features - shape [batch_size, n_features, z', y', x']
        """
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, None, None, None]:  # type: ignore
        """
        Forward pass. 3D global classification given a volume
        :param x: Input volume. shape: [batch_size, channels, z, y, x]
        :return: logits for global classification. shape: [batch_size, n_classes].
        """
        x = self.features(x)
        return x
