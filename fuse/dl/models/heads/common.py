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

from typing import Optional, Sequence
import torch.nn as nn


class ClassifierFCN(nn.Module):
    """
    Sequence of (Conv2D 1X1 , ReLU, Dropout). The length of the sequence and layers size defined by layers_description
    """

    def __init__(
        self,
        in_ch: int,
        num_classes: Optional[int],
        layers_description: Sequence[int] = (256,),
        dropout_rate: float = 0.1,
    ):
        """
        :param in_ch: Number of input channels
        :param num_classes: Appends Conv2D(last_layer_size, num_classes, kernel_size=1, stride=1) if num_classes is not None
        :param layers_description: defines the length of the sequence and layers size.
        :param dropout_rate: if 0 will not include the dropout layers
        """
        super().__init__()
        layer_list = []
        last_layer_size = in_ch

        for i in range(len(layers_description)):
            curr_layer_size = layers_description[i]
            layer_list.append(nn.Conv2d(last_layer_size, curr_layer_size, kernel_size=1, stride=1))
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = curr_layer_size

        if num_classes is not None:
            layer_list.append(nn.Conv2d(last_layer_size, num_classes, kernel_size=1, stride=1))

        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.classifier(x)
        return x


class ClassifierFCN3D(nn.Module):
    """
    Sequence of (Conv3D 1X1 , ReLU, Dropout). The length of the sequence and layers size defined by layers_description
    """

    def __init__(
        self,
        in_ch: int,
        num_classes: Optional[int],
        layers_description: Sequence[int] = (256,),
        dropout_rate: float = 0.1,
    ):
        """
        :param in_ch: Number of input channels
        :param num_classes: Appends Conv2D(last_layer_size, num_classes, kernel_size=1, stride=1) if num_classes is not None
        :param layers_description: defines the length of the sequence and layers size.
        :param dropout_rate: if 0 will not include the dropout layers
        """
        super().__init__()
        layer_list = []
        last_layer_size = in_ch

        for curr_layer_size in layers_description:
            layer_list.append(nn.Conv3d(last_layer_size, curr_layer_size, kernel_size=1, stride=1))
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = curr_layer_size

        if num_classes is not None:
            layer_list.append(nn.Conv3d(last_layer_size, num_classes, kernel_size=1, stride=1))

        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.classifier(x)
        return x


class ClassifierMLP(nn.Module):
    """
    Sequence of (Linear , ReLU, Dropout). The length of the sequence and layers size defined by layers_description
    """

    def __init__(
        self,
        in_ch: int,
        num_classes: Optional[int] = None,
        layers_description: Sequence[int] = (256,),
        dropout_rate: float = 0.1,
        bias: bool = True,
    ):
        """
        :param in_ch: Number of input channels
        :param num_classes: Appends Conv2D(last_layer_size, num_classes, kernel_size=1, stride=1) if num_classes is not None
        :param layers_description: defines the length of the sequence and layers size.
        :param dropout_rate: if 0 will not include the dropout layers
        """
        super().__init__()
        layer_list = []
        last_layer_size = in_ch
        for curr_layer_size in layers_description:
            layer_list.append(nn.Linear(last_layer_size, curr_layer_size))
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = curr_layer_size

        if num_classes is not None:
            layer_list.append(nn.Linear(in_features=last_layer_size, out_features=num_classes, bias=bias))

        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.classifier(x)
        return x
