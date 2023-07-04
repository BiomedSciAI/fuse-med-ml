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

from typing import Sequence, Dict, Tuple, Union

import torch

from fuse.utils.ndict import NDict


class ModelMultiHead(torch.nn.Module):
    """
    Default Fuse model - convolutional neural network with multiple heads
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        heads: Sequence[torch.nn.Module],
        conv_inputs: Union[Sequence[str], Sequence[Tuple[str, int]]] = None,
        backbone_args: Union[Sequence[str], Sequence[Tuple[str, int]]] = None,
        key_out_features: str = "model.backbone_features",
    ) -> None:
        """
        Default Fuse model - convolutional neural network with multiple heads
        :param conv_inputs:     batch_dict name for convolutional backbone model input and its number of input channels. Unused if None. Kept for backward compatibility
        :param backbone_args:   batch_dict name for generic backbone model input and its number of input channels. Unused if None
        :param backbone:        PyTorch backbone module - a convolutional (in which case conv_inputs must be supplied) or some other (in which case backbone_args must be supplied) neural network
        :param heads:           Sequence of head modules
        :param key_out_features: key to where to store the backbone features inside the batch dict
        """
        super().__init__()
        if (conv_inputs is not None) and (backbone_args is not None):
            raise Exception(
                "Both conv_inputs and backbone_args are set. Only one may be set (conv_inputs soon to be deprecated)"
            )
        if (conv_inputs is None) and (backbone_args is None):
            raise Exception(
                "Neither conv_inputs nor backbone_args are None. One must be set (conv_inputs soon to be deprecated)"
            )
        # we don't use the number of input channels.
        # it's just kept for backward compatibility
        if conv_inputs is not None:
            self.conv_inputs = [
                inp[0] if isinstance(inp, Tuple) else inp for inp in conv_inputs
            ]
        if backbone_args is not None:
            self.backbone_args = [
                inp[0] if isinstance(inp, Tuple) else inp for inp in backbone_args
            ]

        self.backbone = backbone
        self.key_out_features = key_out_features
        self.add_module("backbone", self.backbone)
        self.heads = torch.nn.ModuleList(heads)
        self.add_module("heads", self.heads)

    def forward(self, batch_dict: NDict) -> Dict:
        if self.conv_inputs is not None:
            conv_input = torch.cat(
                [batch_dict[conv_input] for conv_input in self.conv_inputs], 1
            )
            backbone_features = self.backbone.forward(conv_input)
        else:
            backbone_args = [batch_dict[inp] for inp in self.backbone_args]
            backbone_features = self.backbone.forward(*backbone_args)

        batch_dict[self.key_out_features] = backbone_features

        for head in self.heads:
            batch_dict = head.forward(batch_dict)

        return batch_dict
