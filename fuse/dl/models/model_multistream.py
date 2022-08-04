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
from typing import Sequence, Dict, Tuple, Callable, Optional

from fuse.utils.ndict import NDict


class ModelMultistream(torch.nn.Module):
    """
    Multi-stream Fuse model - convolutional neural network with multiple processing streams and multiple heads
    """

    def __init__(
        self,
        conv_inputs: Tuple[str, int],
        backbone_streams: Sequence[torch.nn.Module],
        heads: Sequence[torch.nn.Module],
        split_logic: Optional[Callable] = None,
        join_logic: Optional[Callable] = None,
    ) -> None:
        """
        Multi-stream Fuse model - convolutional neural network with multiple processing streams and multiple heads
        :param conv_inputs:             batch_dict name for model input and its number of input channels
            for example: conv_inputs=('data.input.input_0.tensor', 1)
        :param backbone_streams:        List of PyTorch backbone modules - one per stream (can share weights or not)
        :param heads:                   Sequence of head modules
            for example: heads=(HeadGlobalPoolingClassifier(conv_inputs = (('model.backbone_features', 384),))
        :param split_logic:             Optional callable, splits input into streams. If None, sends each input channel to consecutive stream.
                                            Signature: stream_outputs = split_logic(batch_dict, backbone_streams)
        :param join_logic:              Optional callable, joins stream outputs into single feature map. If None, concatenates on channel axis.
                                            Signature: feature_map = join_logic(batch_dict, stream_outputs)
        """
        super().__init__()
        assert (
            conv_inputs is not None
        ), "You must provide conv_inputs - for example: conv_inputs=('data.input.input_0.tensor', 1)"
        self.conv_inputs = conv_inputs
        self.split_logic = split_logic
        self.join_logic = join_logic

        # Register modules
        self.backbone_streams = torch.nn.ModuleList(backbone_streams)
        self.add_module("backbones", self.backbone_streams)
        assert (
            heads is not None
        ), "You must provide heads - for example: heads=(HeadGlobalPoolingClassifier(conv_inputs = (('model.backbone_features', 384),)),)"
        self.heads = torch.nn.ModuleList(heads)
        self.add_module("heads", self.heads)

    def forward(self, batch_dict: NDict) -> Dict:

        # Forward pass through multiple streams
        # -------------------------------------
        if self.split_logic is None:
            # If no split logic is provided, send each channel to different stream
            conv_input = batch_dict[self.conv_inputs[0]]  # shape = [batch_size, num_channels, height, width]
            stream_outputs = []
            for ch_idx in range(conv_input.shape[1]):
                single_channel_batch = conv_input[:, ch_idx, :, :].unsqueeze(
                    dim=1
                )  # shape = [batch_size, 1, height, width]
                stream_output = self.backbone_streams[ch_idx](single_channel_batch)
                stream_outputs.append(stream_output)
        elif callable(self.split_logic):
            stream_outputs = self.split_logic(batch_dict, self.backbone_streams)
        else:
            raise Exception("Error in ModelMultistream - bad split logic provided")

        # Combining feature maps from multiple streams
        # --------------------------------------------
        if self.join_logic is None:
            # If no join logic is provided, concatenate feature maps in channel axis
            backbone_features = torch.cat(stream_outputs, dim=1)
        elif callable(self.join_logic):
            backbone_features = self.join_logic(batch_dict, stream_outputs)
        else:
            raise Exception("Error in ModelMultistream - bad join logic provided")

        batch_dict["model.backbone_features"] = backbone_features
        for head in self.heads:
            batch_dict = head.forward(batch_dict)

        return batch_dict
