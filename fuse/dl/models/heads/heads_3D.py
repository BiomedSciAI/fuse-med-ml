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

from fuse.utils.ndict import NDict
from fuse.dl.models.heads.common import ClassifierFCN3D, ClassifierMLP


class Head3D(nn.Module):
    """
    Model that capture slice feature including the 3D context given the local feature about a slice for classification/regression.
    """

    def __init__(
        self,
        head_name: str = "head_0",
        mode: str = None,  # "classification" or "regression"
        conv_inputs: Sequence[Tuple[str, int]] = None,
        dropout_rate: float = 0.1,
        num_outputs: int = 3,  # num classes in case of classification
        append_features: Optional[Tuple[str, int]] = None,
        layers_description: Sequence[int] = (256,),
        append_layers_description: Sequence[int] = tuple(),
        append_dropout_rate: float = 0.0,
        fused_dropout_rate: float = 0.0,
    ) -> None:
        """
        Create simple 3D context model
        :param head_name: string representing the head name
        :param mode:      "classification" or "regression"
        :param conv_inputs: Sequence of tuples, each indication features name in batch_dict and size of features (channels)
            for example: conv_inputs=(('model.backbone_features', 512),)
            if set to None, the head will work only using the global features.
            can be useful i.e for exploring the contribution of imaging vs. clinical features only.
        :param dropout_rate: dropout fraction
        :param num_outputs:  Number of output classes (in case of classification) or just num outputs in case of regression
        :param append_features: Sequence of tuples, each indication features name in batch_dict and size of features (channels).
                                Those are global features that appended after the global max pooling operation
        :param layers_description:          Layers description for the classifier module - sequence of hidden layers sizes (Not used currently)
        :param append_layers_description: Layers description for the tabular data, before the concatenation with the features extracted from the image - sequence of hidden layers sizes
        :param append_dropout_rate: Dropout rate for tabular layers
        """
        super().__init__()
        # save input params
        self.head_name = head_name
        self.mode = mode
        self.conv_inputs = conv_inputs
        self.dropout_rate = dropout_rate
        self.num_outputs = num_outputs
        self.append_features = append_features
        self.gmp = nn.AdaptiveMaxPool3d(output_size=1)
        self.features_size = sum([features[1] for features in self.conv_inputs]) if self.conv_inputs is not None else 0

        # calc appended feature size if used
        if self.append_features is not None:
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

        self.conv_classifier_3d = ClassifierFCN3D(
            self.features_size, self.num_outputs, layers_description, fused_dropout_rate
        )

        self.do = nn.Dropout3d(p=self.dropout_rate)

    def forward(self, batch_dict: NDict) -> Dict:
        """
        Forward pass
        :param batch_dict: dictionary containing an input tensor representing spatial features with 3D context. shape: [batch_size, in_features, z, y, x]
        :return: batch dict with fields model.outputs and model.logits
        """
        if self.conv_inputs is not None:
            conv_input = torch.cat([batch_dict[conv_input[0]] for conv_input in self.conv_inputs], dim=1)
            global_features = self.gmp(conv_input)
            # backward compatibility
            if hasattr(self, "do"):
                global_features = self.do(global_features)
        # append global features if are used
        if self.append_features is not None:
            features = torch.cat(
                [batch_dict[features[0]].reshape(-1, features[1]) for features in self.append_features], dim=1
            )
            features = self.append_features_module(features)
            features = features.reshape(features.shape + (1, 1, 1))
            if self.conv_inputs is not None:
                global_features = torch.cat((global_features, features), dim=1)
            else:
                global_features = features

        logits = self.conv_classifier_3d(global_features)
        logits = logits.squeeze(dim=4)
        logits = logits.squeeze(dim=3)
        logits = logits.squeeze(dim=2)  # squeeze will change the shape to  [batch_size, channels']
        if self.mode == "regression":
            prediction = logits
            batch_dict["model.output." + self.head_name] = prediction
        else:  # classification
            cls_preds = F.softmax(logits, dim=1)
            batch_dict["model.logits." + self.head_name] = logits
            batch_dict["model.output." + self.head_name] = cls_preds

        return batch_dict
