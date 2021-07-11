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

from typing import Dict, Tuple, Sequence, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fuse.models.heads.common import ClassifierFCN
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseHeadDenseSegmentation(nn.Module):
    def __init__(self,
                 head_name: str = 'head_0',
                 conv_inputs: Sequence[Tuple[str, int]] = (('model.backbone_features', 384),),
                 num_classes: int = 2,
                 post_concat_inputs: Optional[Sequence[Tuple[str, int]]] = None,
                 maxpool_kernel: Optional[Union[Tuple[int, int], int]] = None,
                 layers_description: Sequence[int] = (256,),
                 dropout_rate: float = 0.1,
                 shared_classifier_head: Optional[torch.nn.Module] = None,
                 ) -> None:
        """
        Dense segmentation head - predicts class scores for each location of input feature map ("conv_input").

        Output of a forward pass:
        'model.logits.head_name' and 'outputs.head_name', both in shape [batch_size, num_classes, input_height, input_width]

        :param head_name:                   batch_dict key
        :param conv_inputs:                 List of feature map inputs - tuples of (batch_dict key, channel depth)
                                            If multiple inputs are used, they are concatenated on the channel axis
        :param num_classes:                 Number of output classes (per feature map location)
        :param post_concat_inputs:          Additional vector (one dimensional) inputs, concatenated just before the classifier module
        :param maxpool_kernel:              Kernel size for an optional preliminary max pooling step, to reduce feature maps size
        :param layers_description:          Layers description for the classifier module - sequence of hidden layers sizes
        :param dropout_rate:                Dropout rate for classifier module layers
        :param shared_classifier_head:      Optional reference for external torch.nn.Module classifier
        """
        super().__init__()

        self.head_name = head_name
        self.conv_inputs = conv_inputs
        self.maxpool_kernel = maxpool_kernel

        feature_depth = sum([conv_input[1] for conv_input in self.conv_inputs])
        if post_concat_inputs is not None:
            feature_depth += sum([post_concat_input[1] for post_concat_input in post_concat_inputs])

        if shared_classifier_head is not None:
            self.classifier_head_module = shared_classifier_head
        else:
            self.classifier_head_module = ClassifierFCN(in_ch=feature_depth,
                                                        num_classes=num_classes,
                                                        layers_description=layers_description,
                                                        dropout_rate=dropout_rate)

    def forward(self,
                batch_dict: Dict) -> Dict:
        conv_input = torch.cat([FuseUtilsHierarchicalDict.get(batch_dict, conv_input[0]) for conv_input in self.conv_inputs])
        if self.maxpool_kernel is not None:
            conv_input = F.max_pool2d(conv_input, kernel_size=self.maxpool_kernel)

        logits = self.classifier_head_module(conv_input)
        score_map = F.softmax(logits, dim=1)  # --> score_map.shape = [batch_size, 2, height, width]

        FuseUtilsHierarchicalDict.set(batch_dict, 'model.logits.' + self.head_name, logits)
        FuseUtilsHierarchicalDict.set(batch_dict, 'model.output.' + self.head_name, score_map)

        return batch_dict
