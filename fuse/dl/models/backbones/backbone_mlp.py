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

from typing import Dict, List

import torch
import torch.nn as nn
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseMultilayerPerceptronBackbone(torch.nn.Module):

    def __init__(self,
                 layers: List[int] = (64, 192, 320, 320, 1088, 384),
                 mlp_input_size: int = 103,
                 activation_layer: torch.nn.Module = nn.ReLU(inplace=False),
                 dropout_rate: float = 0.0,
                 ) -> None:
        super().__init__()

        mlp_layers = [nn.Linear(mlp_input_size, layers[0])]
        if activation_layer is not None:
            mlp_layers.append(activation_layer)
        if dropout_rate > 0:
            mlp_layers.append(nn.Dropout(p=dropout_rate))

        for layer_idx in range(len(layers) - 1):
            mlp_layers.append(nn.Linear(layers[layer_idx], layers[layer_idx + 1]))
            mlp_layers.append(activation_layer) if activation_layer is not None else None
            mlp_layers.append(nn.Dropout(p=dropout_rate)) if dropout_rate > 0 else None

        self.mlp = nn.ModuleList(mlp_layers)

    def forward(self, input_tensor: Dict) -> Dict:
        for layer in self.mlp:
            input_tensor = layer(input_tensor)
        return input_tensor
