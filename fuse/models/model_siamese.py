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

from fuse.models.model_default import FuseModelDefault
from fuse.models.backbones.backbone_inception_resnet_v2 import FuseBackboneInceptionResnetV2
from fuse.models.heads.head_global_pooling_classifier import FuseHeadGlobalPoolingClassifier
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseModelSiamese(FuseModelDefault):
    """
    Fuse Siamese model - 2 branches of the same convolutional neural network with multiple heads
    """

    def __init__(self,
                 conv_inputs_0: Tuple[Tuple[str, int], ...] = (('data.input.input_0.tensor', 1),),
                 conv_inputs_1: Tuple[Tuple[str, int], ...] = (('data.input.input_1.tensor', 1),),
                 backbone: torch.nn.Module = FuseBackboneInceptionResnetV2(),
                 heads: Sequence[torch.nn.Module] = (FuseHeadGlobalPoolingClassifier(),)
                 ) -> None:
        """
        Fuse Siamese model -  two branches with same convolutional neural network with multiple heads
        :param conv_inputs:     batch_dict name for model input and its number of input channels
        :param backbone:        PyTorch backbone module - a convolutional neural network
        :param heads:           Sequence of head modules
            """
        super().__init__((), backbone, heads)
        self.conv_inputs_0 = conv_inputs_0
        self.conv_inputs_1 = conv_inputs_1

    def forward(self,
                batch_dict: Dict) -> Dict:
        backbone_input_0 = [FuseUtilsHierarchicalDict.get(batch_dict, conv_input[0]) for conv_input in self.conv_inputs_0]  #[batch,channels(1),h,w]
        backbone_input_1 = [FuseUtilsHierarchicalDict.get(batch_dict, conv_input[0]) for conv_input in self.conv_inputs_1]  #[batch,channels(1),h,w]

        backbone_features_0 = self.backbone.forward(torch.stack(backbone_input_0, dim=1))  #batch, features, h', w'
        backbone_features_1 = self.backbone.forward(torch.stack(backbone_input_1, dim=1))  #batch, features, h', w'

        backbone_features = torch.cat([backbone_features_0, backbone_features_1], dim=1)

        FuseUtilsHierarchicalDict.set(batch_dict, 'model.backbone_features', backbone_features)
        FuseUtilsHierarchicalDict.set(batch_dict, 'model.backbone_features_0', backbone_features_0)
        FuseUtilsHierarchicalDict.set(batch_dict, 'model.backbone_features_1', backbone_features_1)

        for head in self.heads:
            batch_dict = head.forward(batch_dict)

        return batch_dict['model']
