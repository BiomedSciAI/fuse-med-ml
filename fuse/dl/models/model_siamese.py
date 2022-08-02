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

from typing import Sequence, Dict, Tuple

import torch

from fuse.dl.models import ModelMultiHead
from fuse.utils.ndict import NDict


class ModelSiamese(ModelMultiHead):
    """
    Fuse Siamese model - 2 branches of the same convolutional neural network with multiple heads
    """

    def __init__(
        self,
        conv_inputs_0: Tuple[Tuple[str, int], ...],
        conv_inputs_1: Tuple[Tuple[str, int], ...],
        backbone: torch.nn.Module,
        heads: Sequence[torch.nn.Module] = None,
    ) -> None:
        """
        Fuse Siamese model -  two branches with same convolutional neural network with multiple heads
        :param conv_inputs_0:     batch_dict name for model input and its number of input channels
            for example: conv_inputs_0=(('data.input.input_0.tensor', 1),)
        :param conv_inputs_1:    batch_dict name for model input and its number of input channels
            for example: conv_inputs_1=(('data.input.input_1.tensor', 1),)
        :param backbone:        PyTorch backbone module - a convolutional neural network
        :param heads:           Sequence of head modules
            for example: (HeadGlobalPoolingClassifier(),)
        """
        assert heads is not None, "You must provide 'heads'"
        super().__init__((), backbone, heads)
        assert conv_inputs_0 is not None, "you must provide 'conv_inputs_0'"
        self.conv_inputs_0 = conv_inputs_0
        assert conv_inputs_1 is not None, "you must provide 'conv_inputs_1'"
        self.conv_inputs_1 = conv_inputs_1

    def forward(self, batch_dict: NDict) -> Dict:
        backbone_input_0 = [batch_dict[conv_input[0]] for conv_input in self.conv_inputs_0]  # [batch,channels(1),h,w]
        backbone_input_1 = [batch_dict[conv_input[0]] for conv_input in self.conv_inputs_1]  # [batch,channels(1),h,w]

        backbone_features_0 = self.backbone.forward(torch.stack(backbone_input_0, dim=1))  # batch, features, h', w'
        backbone_features_1 = self.backbone.forward(torch.stack(backbone_input_1, dim=1))  # batch, features, h', w'

        backbone_features = torch.cat([backbone_features_0, backbone_features_1], dim=1)

        batch_dict["model.backbone_features"] = backbone_features
        batch_dict["model.backbone_features_0"] = backbone_features_0
        batch_dict["model.backbone_features_1"] = backbone_features_1

        for head in self.heads:
            batch_dict = head.forward(batch_dict)

        return batch_dict
