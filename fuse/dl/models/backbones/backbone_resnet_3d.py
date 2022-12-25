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
from torchvision.models.video import mc3_18, r3d_18


class BackboneResnet3D(nn.Module):
    """
    3D model classifier (ResNet architecture"
    """

    def __init__(self, pretrained: bool = False, in_channels: int = 3, name: str = "r3d_18") -> None:
        """
        Create 3D ResNet model
        :param pretrained: Use pretrained weights
        :param in_channels: number of input channels
        :param name: model name. currently only 'r3d_18' is supported
        """
        # init original model
        super().__init__()
        if name == "r3d_18":
            self.model = r3d_18(pretrained=pretrained)
        elif name == "mc3_18":
            self.model = mc3_18(pretrained=pretrained)

        

        if in_channels != 3:
            # override the first convolution layer to support any number of input channels
            self.model.stem = nn.Sequential(
                nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
            )
        self.model = nn.Sequential(self.model.stem, self.model.layer1,self.model.layer2, self.model.layer3, self.model.layer4)

    def forward(self, x: Tensor) -> Tuple[Tensor, None, None, None]:  # type: ignore
        """
        Forward pass. 3D global classification given a volume
        :param x: Input volume. shape: [batch_size, channels, z, y, x]
        :return: logits for global classification. shape: [batch_size, n_classes].
        """
        x = self.model(x)
        return x
