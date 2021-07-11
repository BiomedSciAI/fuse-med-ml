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

from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseHead3dClassifier(nn.Module):
    """
    Model that capture slice feature including the 3D context given the local feature about a slice.
    """

    def __init__(self, head_name: str = 'head_0',
                 conv_inputs: Sequence[Tuple[str, int]] = (('model.backbone_features', 512),),
                 dropout_rate: float = 0.1,
                 num_classes: int = 3,
                 append_features: Optional[Tuple[str, int]] = None
                 ) -> None:
        """
        Create simple 3D context model
        :param head_name: string representing the head name
        :param conv_inputs: Sequence of tuples, each indication features name in batch_dict and size of features (channels)
        :param dropout_rate: dropout fraction
        :param num_classes: number of output classes
        :param append_features: Sequence of tuples, each indication features name in batch_dict and size of features (channels).
                                Those are global features that appended after the global max pooling operation

        """
        super().__init__()
        # save input params
        self.head_name = head_name
        self.conv_inputs = conv_inputs
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.append_features = append_features
        self.gmp = nn.AdaptiveMaxPool3d(output_size=1)
        self.features_size = sum([features[1] for features in self.conv_inputs]) if self.conv_inputs is not None else 0

        # calc appended feature size if used
        if self.append_features is not None:
            global_features_size = sum([features[1] for features in self.append_features])
            self.features_size += global_features_size

        self.conv_classifier_3d = nn.Sequential(
            nn.Conv3d(self.features_size, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout3d(p=self.dropout_rate),
            nn.Conv3d(256, self.num_classes, kernel_size=1),
        )

        self.do = nn.Dropout3d(p=self.dropout_rate)

    def forward(self, batch_dict: Dict) -> Dict:
        """
        Forward pass
        :param batch_dict: dictionary containing an input tensor representing spatial features with 3D context. shape: [batch_size, in_features, z, y, x]
        :return: batch dict with fields model.outputs and model.logits
        """
        if self.conv_inputs is not None:
            conv_input = torch.cat(
                [FuseUtilsHierarchicalDict.get(batch_dict, conv_input[0]) for conv_input in self.conv_inputs], dim=1)
            global_features = self.gmp(conv_input)
            # save global max pooling features in case needed (mostly to analyze)
            FuseUtilsHierarchicalDict.set(batch_dict, 'model.' + self.head_name +'.gmp_features', global_features.squeeze(dim=4).squeeze(dim=3).squeeze(dim=2))
            # backward compatibility
            if hasattr(self, 'do'):
                global_features = self.do(global_features)
        # append global features if are used
        if self.append_features is not None:
            features = torch.cat(
                [FuseUtilsHierarchicalDict.get(batch_dict, features[0]).reshape(-1, features[1]) for features in self.append_features], dim=1)
            features = features.reshape(features.shape + (1,1,1))
            if self.conv_inputs is not None:
                global_features = torch.cat((global_features, features), dim=1)
            else:
                global_features = features

        logits = self.conv_classifier_3d(global_features)
        logits = logits.squeeze(dim=4)
        logits = logits.squeeze(dim=3)
        logits = logits.squeeze(dim=2)  # squeeze will change the shape to  [batch_size, channels']

        cls_preds = F.softmax(logits, dim=1)
        FuseUtilsHierarchicalDict.set(batch_dict, 'model.logits.' + self.head_name, logits)
        FuseUtilsHierarchicalDict.set(batch_dict, 'model.output.' + self.head_name, cls_preds)

        return batch_dict
