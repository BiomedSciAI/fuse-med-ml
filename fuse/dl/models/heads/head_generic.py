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

from typing import Dict, Tuple, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fuse.dl.models.heads.common import ClassifierFCN, ClassifierMLP
from fuse.utils.ndict import NDict


class HeadGeneric(nn.Module):
    def __init__(
        self,
        head: torch.nn.Module,
        head_name: str = "head_0",
        conv_inputs: Sequence[Tuple[str, int]] = None,
        dropout_rate: float = 0.1,
    ) -> None:
        """

        Output of a forward pass:
        'model.head_name' in shape [batch_size, num_classes]

        :param head_name:                   batch_dict key
        :param conv_inputs:                 List of feature map inputs - tuples of (batch_dict key, channel depth)
                                            If multiple inputs are used, they are concatenated on the channel axis
                example: conv_inputs = (('model.backbone_features', 384),)
        :param dropout_rate:                Dropout rate for classifier module layers
        :param head:      reference for external torch.nn.Module classifier
        """
        super().__init__()

        self.head_name = head_name
        assert conv_inputs is not None, "conv_inputs must be provided"
        self.conv_inputs = conv_inputs
        self.head = head

    def forward(self, batch_dict: NDict) -> Dict:
        res = torch.cat([batch_dict[conv_input[0]] for conv_input in self.conv_inputs])

        out = self.head(res)

        batch_dict["model." + self.head_name] = out
        return batch_dict
