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

try:
    from torch.hub import load_state_dict_from_url
except:
    pass
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, model_urls, BasicBlock
from torch import Tensor


class FuseBackboneResnet(ResNet):
    """
    2D ResNet backbone
    """

    def __init__(self, pretrained: bool = False, in_channels: int = 2, name: str = "resnet18") -> None:
        """
        Create 2D Resnet
        :param pretrained: reload imagenet weights
        :param in_channels: Number of input channels
        :param name: model name. Currently support 'resnet18' and 'resnet50'
        """
        # init parameters per required backbone
        init_parameters = {
            "resnet18": [BasicBlock, [2, 2, 2, 2]],
            "resnet50": [Bottleneck, [3, 4, 6, 3]]
        }[name]
        # init original model
        super().__init__(*init_parameters)

        # load pretrained parameters if required
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[name])
            self.load_state_dict(state_dict)

        # save input parameters
        self.pretrained = pretrained
        self.in_channels = in_channels

        # override the first convolution layer to support any number of input channels
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        """
        Forward pass extract spatial features
        :param x: input tensor. Shape [batch_size, input_channels, y, x]
        :return: spatial features [batch_size, n_features, y', x']
        """
        # x assumed to be [batch_size, in_channels, y, x]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
