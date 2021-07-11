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


class ClassifierFCN(nn.Module):
    def __init__(self, in_ch, num_classes, layers_description=(256,), dropout_rate=0.1):
        super(ClassifierFCN, self).__init__()
        layer_list = []
        layer_list.append(nn.Conv2d(in_ch, layers_description[0], kernel_size=1, stride=1))
        layer_list.append(nn.ReLU())
        if dropout_rate is not None and dropout_rate > 0:
            layer_list.append(nn.Dropout(p=dropout_rate))
        last_layer_size = layers_description[0]
        for curr_layer_size in layers_description[1:]:
            layer_list.append(nn.Conv2d(last_layer_size, curr_layer_size, kernel_size=1, stride=1))
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = curr_layer_size

        layer_list.append(nn.Conv2d(last_layer_size, num_classes, kernel_size=1, stride=1))
        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.classifier(x)
        return x
