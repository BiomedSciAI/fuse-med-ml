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


class HeadGlobalPoolingClassifier(nn.Module):
    def __init__(
        self,
        head_name: str = "head_0",
        conv_inputs: Sequence[Tuple[str, int]] = None,
        num_classes: int = 2,
        tabular_data_inputs: Optional[Sequence[Tuple[str, int]]] = None,
        pooling: str = "max",
        layers_description: Sequence[int] = (256,),
        tabular_layers_description: Sequence[int] = tuple(),
        dropout_rate: float = 0.1,
        tabular_dropout_rate: float = 0.0,
        shared_classifier_head: Optional[torch.nn.Module] = None,
    ) -> None:
        """
        Classifier head with a global pooling operator.

        Output of a forward pass:
        'model.logits.head_name' and 'model.output.head_name', both in shape [batch_size, num_classes]

        :param head_name:                   batch_dict key
        :param conv_inputs:                 List of feature map inputs - tuples of (batch_dict key, channel depth)
                                            If multiple inputs are used, they are concatenated on the channel axis
                example: conv_inputs = (('model.backbone_features', 384),)
        :param num_classes:                 Number of output classes (per feature map location)
        :param tabular_data_inputs:          Additional vector (one dimensional) inputs, concatenated just before the classifier module
        :param pooling:                     Type of global pooling operator ('max' or 'avg')
        :param layers_description:          Layers description for the classifier module - sequence of hidden layers sizes
        :param tabular_layers_description: Layers description for the tabular data, before the concatination with the features extracted from the image - sequence of hidden layers sizes
        :param dropout_rate:                Dropout rate for classifier module layers
        :param tabular_dropout_rate:        Dropout rate for tabular layers
        :param shared_classifier_head:      Optional reference for external torch.nn.Module classifier
        """
        super().__init__()

        assert pooling in ("max", "avg")

        self.head_name = head_name
        assert conv_inputs is not None, "conv_inputs must be provided"
        self.conv_inputs = conv_inputs
        self.tabular_data_inputs = tabular_data_inputs
        self.pooling = pooling

        feature_depth = sum([conv_input[1] for conv_input in self.conv_inputs])

        if tabular_data_inputs is not None:
            if len(tabular_layers_description) == 0:
                feature_depth += sum([post_concat_input[1] for post_concat_input in tabular_data_inputs])
                self.tabular_module = nn.Identity()
            else:
                feature_depth += tabular_layers_description[-1]
                self.tabular_module = ClassifierMLP(
                    in_ch=sum([post_concat_input[1] for post_concat_input in tabular_data_inputs]),
                    num_classes=None,
                    layers_description=tabular_layers_description,
                    dropout_rate=tabular_dropout_rate,
                )

        if shared_classifier_head is not None:
            self.classifier_head_module = shared_classifier_head
        else:
            self.classifier_head_module = ClassifierFCN(
                in_ch=feature_depth,
                num_classes=num_classes,
                layers_description=layers_description,
                dropout_rate=dropout_rate,
            )

    def forward(self, batch_dict: NDict) -> Dict:
        conv_input = torch.cat([batch_dict[conv_input[0]] for conv_input in self.conv_inputs])

        if len(conv_input.shape) == 2:
            res = conv_input
        else:
            # Global pooling - input shape is [batch_size, num_channels, height, width], result is [batch_size, num_channels, 1, 1]
            if self.pooling == "max":
                res = F.max_pool2d(conv_input, kernel_size=conv_input.shape[2:])
            elif self.pooling == "avg":
                res = F.avg_pool2d(conv_input, kernel_size=conv_input.shape[2:])

        if self.tabular_data_inputs is not None:
            tabular_input = torch.cat([batch_dict[tabular_input[0]] for tabular_input in self.tabular_data_inputs])
            tabular_input = self.tabular_module(tabular_input)
            tabular_input = tabular_input.reshape(tabular_input.shape + (1, 1))
            res = torch.cat([res, tabular_input], dim=1)

        logits = self.classifier_head_module(res)  # --> res.shape = [batch_size, 2, 1, 1]
        if len(logits.shape) > 2:
            logits = logits.squeeze(dim=3)  # --> res.shape = [batch_size, 2, 1]
            logits = logits.squeeze(dim=2)  # --> res.shape = [batch_size, 2]

        cls_preds = F.softmax(logits, dim=1)

        batch_dict["model.logits." + self.head_name] = logits
        batch_dict["model.output." + self.head_name] = cls_preds

        return batch_dict
