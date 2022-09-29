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

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple, Sequence, Optional
from fuse.dl.models.heads.common import ClassifierMLP


class Head1D(nn.Module):
    def __init__(
        self,
        head_name: str = "head_0",
        mode: str = None,  # "classification" or "regression"
        conv_inputs: Sequence[Tuple[str, int]] = None,
        num_outputs: int = 2,  # num classes in case of classification
        append_features: Optional[Sequence[Tuple[str, int]]] = None,
        layers_description: Sequence[int] = (256,),
        append_layers_description: Sequence[int] = tuple(),
        append_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
    ) -> None:
        """
        head 1d.

        Output of a forward pass for classification:
        'model.logits.head_name' and 'outputs.head_name', both in shape [batch_size, num_outputs]
        Output of a forward pass for regression:
        'model.output.head_name' in shape [batch_size, num_outputs]

        :param head_name:                   batch_dict key
        :param mode:                        "classification" or "regression"
        :param conv_inputs:                 List of feature map inputs - tuples of (batch_dict key, channel depth)
                                            If multiple inputs are used, they are concatenated on the channel axis
                for example:
                conv_inputs=(('model.backbone_features', 193),)
        :param num_outputs:                 Number of output classes (in case of classification) or just num outputs in case of regression
        :param append_features:          Additional vector (one dimensional) inputs, concatenated just before the classifier module
        :param layers_description:          Layers description for the classifier module - sequence of hidden layers sizes
        :param dropout_rate:                Dropout rate for classifier module layers
        """
        super().__init__()

        self.head_name = head_name
        self.mode = mode
        assert conv_inputs is not None, "conv_inputs must be provided"
        self.conv_inputs = conv_inputs
        self.append_features = append_features

        self.features_size = sum([conv_input[1] for conv_input in self.conv_inputs])

        if append_features is not None:
            if len(append_layers_description) == 0:
                self.features_size += sum([post_concat_input[1] for post_concat_input in append_features])
                self.append_features_module = nn.Identity()
            else:
                self.features_size += append_layers_description[-1]
                self.append_features_module = ClassifierMLP(
                    in_ch=sum([post_concat_input[1] for post_concat_input in append_features]),
                    num_classes=None,
                    layers_description=append_layers_description,
                    dropout_rate=append_dropout_rate,
                )

        self.head_module = ClassifierMLP(
            in_ch=self.features_size,
            num_classes=num_outputs,
            layers_description=layers_description,
            dropout_rate=dropout_rate,
        )

    def forward(self, batch_dict: Dict) -> Dict:

        conv_input = torch.cat([batch_dict[conv_input[0]] for conv_input in self.conv_inputs], dim=1)
        global_features = conv_input

        if self.append_features is not None:
            features = torch.cat([batch_dict[append_feature[0]] for append_feature in self.append_features])
            features = self.append_features_module(features)
            features = features.reshape(features.shape + (1, 1, 1))
            if self.conv_inputs is not None:
                global_features = torch.cat((global_features, features), dim=1)
            else:
                global_features = features
        if self.mode == "regression":
            prediction = self.head_module(global_features).squeeze(dim=1)
            batch_dict["model.output." + self.head_name] = prediction
        else:
            logits = self.head_module(global_features)  # --> res.shape = [batch_size, 2, 1, 1]
            if len(logits.shape) > 2:
                logits = logits.squeeze(dim=3)  # --> res.shape = [batch_size, 2, 1]
                logits = logits.squeeze(dim=2)  # --> res.shape = [batch_size, 2]

            cls_preds = F.softmax(logits, dim=1)

            batch_dict["model.logits." + self.head_name] = logits
            batch_dict["model.output." + self.head_name] = cls_preds

        return batch_dict
