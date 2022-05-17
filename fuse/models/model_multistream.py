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

from typing import Sequence, Dict, Tuple, Callable, Optional

import torch

from fuse.models.backbones.backbone_inception_resnet_v2 import FuseBackboneInceptionResnetV2
from fuse.models.heads.head_global_pooling_classifier import FuseHeadGlobalPoolingClassifier
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseModelMultistream(torch.nn.Module):
    """
    Multi-stream Fuse model - convolutional neural network with multiple processing streams and multiple heads
    """

    def __init__(self,
                 conv_inputs: Tuple[str, int] = ('data.input.input_0.tensor', 1),
                 backbone_streams: Sequence[torch.nn.Module] = (FuseBackboneInceptionResnetV2(logical_units_num=12),
                                                                FuseBackboneInceptionResnetV2(logical_units_num=12)),
                 heads: Sequence[torch.nn.Module] = (FuseHeadGlobalPoolingClassifier(),),
                 split_logic: Optional[Callable] = None,
                 join_logic: Optional[Callable] = None,
                 ) -> None:
        """
        Multi-stream Fuse model - convolutional neural network with multiple processing streams and multiple heads
        :param conv_inputs:             batch_dict name for model input and its number of input channels
        :param backbone_streams:        List of PyTorch backbone modules - one per stream (can share weights or not)
        :param heads:                   Sequence of head modules
        :param split_logic:             Optional callable, splits input into streams. If None, sends each input channel to consecutive stream.
                                            Signature: stream_outputs = split_logic(batch_dict, backbone_streams)
        :param join_logic:              Optional callable, joins stream outputs into single feature map. If None, concatenates on channel axis.
                                            Signature: feature_map = join_logic(batch_dict, stream_outputs)
        """
        super().__init__()
        self.conv_inputs = conv_inputs
        self.split_logic = split_logic
        self.join_logic = join_logic

        # Register modules
        self.backbone_streams = torch.nn.ModuleList(backbone_streams)
        self.add_module('backbones', self.backbone_streams)
        self.heads = torch.nn.ModuleList(heads)
        self.add_module('heads', self.heads)

    def forward(self,
                batch_dict: Dict) -> Dict:

        # Forward pass through multiple streams
        # -------------------------------------
        if self.split_logic is None:
            # If no split logic is provided, send each channel to different stream
            conv_input = FuseUtilsHierarchicalDict.get(batch_dict, self.conv_inputs[0])  # shape = [batch_size, num_channels, height, width]
            stream_outputs = []
            for ch_idx in range(conv_input.shape[1]):
                single_channel_batch = conv_input[:, ch_idx, :, :].unsqueeze(dim=1)  # shape = [batch_size, 1, height, width]
                stream_output = self.backbone_streams[ch_idx](single_channel_batch)
                stream_outputs.append(stream_output)
        elif callable(self.split_logic):
            stream_outputs = self.split_logic(batch_dict, self.backbone_streams)
        else:
            raise Exception('Error in FuseModelMultistream - bad split logic provided')

        # Combining feature maps from multiple streams
        # --------------------------------------------
        if self.join_logic is None:
            # If no join logic is provided, concatenate feature maps in channel axis
            backbone_features = torch.cat(stream_outputs, dim=1)
        elif callable(self.join_logic):
            backbone_features = self.join_logic(batch_dict, stream_outputs)
        else:
            raise Exception('Error in FuseModelMultistream - bad join logic provided')

        FuseUtilsHierarchicalDict.set(batch_dict, 'model.backbone_features', backbone_features)
        for head in self.heads:
            batch_dict = head.forward(batch_dict)

        return batch_dict['model']


if __name__ == '__main__':
    from fuse.models.heads.head_dense_segmentation import FuseHeadDenseSegmentation

    backbone_0 = FuseBackboneInceptionResnetV2(logical_units_num=8)
    backbone_1 = FuseBackboneInceptionResnetV2(logical_units_num=8)

    non_shared_model = FuseModelMultistream(
        conv_inputs=('data.input.input_0.tensor', 2),
        backbone_streams=[backbone_0, backbone_1],
        heads=[
            FuseHeadGlobalPoolingClassifier(head_name='head_0',
                                            conv_inputs=[('model.backbone_features', 640)],
                                            post_concat_inputs=None,
                                            num_classes=2),

            FuseHeadDenseSegmentation(head_name='head_1',
                                      conv_inputs=[('model.backbone_features', 640)],
                                      num_classes=2)
        ]
    )

    shared_model = FuseModelMultistream(
        conv_inputs=('data.input.input_0.tensor', 2),
        backbone_streams=[backbone_0, backbone_0],
        heads=[
            FuseHeadGlobalPoolingClassifier(head_name='head_0',
                                            conv_inputs=[('model.backbone_features', 640)],
                                            post_concat_inputs=None,
                                            num_classes=2),

            FuseHeadDenseSegmentation(head_name='head_1',
                                      conv_inputs=[('model.backbone_features', 640)],
                                      num_classes=2)
        ]
    )

    dummy_data = {'data':
                      {'input':
                           {'input_0': {'tensor': torch.zeros([20, 2, 200, 100]), 'metadata': None}},
                       'gt':
                           {'gt_0': torch.zeros([0])}
                       }
                  }

    for model_name, model in [('Shared-model', shared_model), ('Non-shared-model', non_shared_model)]:
        res = {}
        res['model'] = shared_model.forward(dummy_data)
        print(model_name + ', forward pass shape - head_0: ', end='')
        print(str(res['model']['logits']['head_0'].shape))
        print(model_name + ', forward pass shape - head_1: ', end='')
        print(str(res['model']['logits']['head_1'].shape))
        total_params = sum(p.numel() for p in model.parameters())
        backbone_params = sum(p.numel() for p in model._modules['backbones'].parameters())
        print(model_name + ', backbone params = %d' % backbone_params)
        print(model_name + ', heads params = %d' % (total_params - backbone_params))
        print(model_name + ', total params = %d' % total_params)
        print()
