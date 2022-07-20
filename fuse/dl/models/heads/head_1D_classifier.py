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


class ClassifierLinear(nn.Module):
    def __init__(
        self,
        in_ch: Sequence[int] = (256,),
        num_classes: float = 2,
        layers_description: Sequence[int] = (256,),
        dropout_rate: float = 0.1,
    ):

        super(ClassifierLinear, self).__init__()

        layer_list = []
        if layers_description is not None:
            layer_list.append(nn.Linear(in_ch, layers_description[0]))
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = layers_description[0]
            if len(layers_description) > 1:
                for curr_layer_size in layers_description[1:]:
                    layer_list.append(nn.Linear(last_layer_size, curr_layer_size))
                    layer_list.append(nn.ReLU())
                    if dropout_rate is not None and dropout_rate > 0:
                        layer_list.append(nn.Dropout(p=dropout_rate))
                    last_layer_size = curr_layer_size
        else:
            last_layer_size = in_ch

        layer_list.append(nn.Linear(last_layer_size, num_classes))
        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.classifier(x)
        return x


class LinearLayers(nn.Module):
    def __init__(
        self,
        in_ch: Sequence[int] = (256,),
        layers_description: Sequence[int] = (256,),
        dropout_rate: float = 0.1,
    ):

        super(LinearLayers, self).__init__()
        layer_list = []
        layer_list.append(nn.Linear(in_ch, layers_description[0]))
        layer_list.append(nn.ReLU())
        if dropout_rate is not None and dropout_rate > 0:
            layer_list.append(nn.Dropout(p=dropout_rate))
        last_layer_size = layers_description[0]
        if len(layers_description) > 1:
            for curr_layer_size in layers_description[1:]:
                layer_list.append(nn.Linear(last_layer_size, curr_layer_size))
                layer_list.append(nn.ReLU())
                if dropout_rate is not None and dropout_rate > 0:
                    layer_list.append(nn.Dropout(p=dropout_rate))
                last_layer_size = curr_layer_size

        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.classifier(x)
        return x


class Head1DClassifier(nn.Module):
    def __init__(
        self,
        head_name: str = "head_0",
        conv_inputs: Sequence[Tuple[str, int]] = None,
        num_classes: int = 2,
        post_concat_inputs: Optional[Sequence[Tuple[str, int]]] = None,
        post_concat_model: Optional[Sequence[int]] = None,
        layers_description: Sequence[int] = (256,),
        dropout_rate: float = 0.1,
        shared_classifier_head: Optional[torch.nn.Module] = None,
    ) -> None:
        """
        Classifier head 1d.

        Output of a forward pass:
        'model.logits.head_name' and 'outputs.head_name', both in shape [batch_size, num_classes]

        :param head_name:                   batch_dict key
        :param conv_inputs:                 List of feature map inputs - tuples of (batch_dict key, channel depth)
                                            If multiple inputs are used, they are concatenated on the channel axis
                for example:
                conv_inputs=(('model.backbone_features', 193),)
        :param num_classes:                 Number of output classes (per feature map location)
        :param post_concat_inputs:          Additional vector (one dimensional) inputs, concatenated just before the classifier module
        :param post_concat_model            Layers description for the post_concat_inputs module - sequence of hidden layers sizes
        :param layers_description:          Layers description for the classifier module - sequence of hidden layers sizes
        :param dropout_rate:                Dropout rate for classifier module layers
        :param shared_classifier_head:      Optional reference for external torch.nn.Module classifier
        """
        super().__init__()

        self.head_name = head_name
        assert conv_inputs is not None, "conv_inputs must be provided"
        self.conv_inputs = conv_inputs
        self.post_concat_inputs = post_concat_inputs
        self.post_concat_model = post_concat_model

        feature_depth = sum([conv_input[1] for conv_input in self.conv_inputs])

        if (post_concat_inputs is not None) and (self.post_concat_model is None):
            ## concat post_concat_input directly to conv_input,
            feature_depth += sum([post_concat_input[1] for post_concat_input in post_concat_inputs])

        elif self.post_concat_model is not None:
            # concat post_concat_input features from classifier_post_concat_model to conv_input
            features_depth_post_concat = sum([post_concat_input[1] for post_concat_input in post_concat_inputs])
            self.classifier_post_concat_model = LinearLayers(
                in_ch=features_depth_post_concat, layers_description=self.post_concat_model, dropout_rate=dropout_rate
            )

        if shared_classifier_head is not None:
            self.classifier_head_module = shared_classifier_head
        else:
            self.classifier_head_module = ClassifierLinear(
                in_ch=feature_depth,
                num_classes=num_classes,
                layers_description=layers_description,
                dropout_rate=dropout_rate,
            )

    def forward(self, batch_dict: Dict) -> Dict:

        conv_input = torch.cat([batch_dict[conv_input[0]] for conv_input in self.conv_inputs])

        res = conv_input

        if self.post_concat_inputs is not None:
            post_concat_input = torch.cat(
                [batch_dict[post_concat_input[0]] for post_concat_input in self.post_concat_inputs]
            )
            if self.post_concat_model is None:
                # concat post_concat_input directly to conv_input
                res = torch.cat([res, post_concat_input], dim=1)
            else:
                # concat post_concat_input features from classifier_post_concat_model to conv_input
                post_concat_input_feat = self.classifier_post_concat_model(post_concat_input)
                res = torch.cat([res, post_concat_input_feat], dim=1)

        logits = self.classifier_head_module(res)  # --> res.shape = [batch_size, 2, 1, 1]
        if len(logits.shape) > 2:
            logits = logits.squeeze(dim=3)  # --> res.shape = [batch_size, 2, 1]
            logits = logits.squeeze(dim=2)  # --> res.shape = [batch_size, 2]

        cls_preds = F.softmax(logits, dim=1)

        batch_dict["model.logits." + self.head_name] = logits
        batch_dict["model.output." + self.head_name] = cls_preds

        return batch_dict
