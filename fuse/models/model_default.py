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

from fuse.models.backbones.backbone_inception_resnet_v2 import FuseBackboneInceptionResnetV2
from fuse.models.heads.head_global_pooling_classifier import FuseHeadGlobalPoolingClassifier
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseModelDefault(torch.nn.Module):
    """
    Default Fuse model - convolutional neural network with multiple heads
    """

    def __init__(self,

                 backbone: torch.nn.Module,
                 heads: Sequence[torch.nn.Module],
                 conv_inputs: Tuple[Tuple[str, int], ...]=None,
                 backbone_args: Tuple[Tuple[str, int], ...]=None,
                 ) -> None:
        """
        Default Fuse model - convolutional neural network with multiple heads
        :param conv_inputs:     batch_dict name for convolutional backbone model input and its number of input channels. Unused if None. Kept for backward compatibility
        :param backbone_args:   batch_dict name for generic backbone model input and its number of input channels. Unused if None
        :param backbone:        PyTorch backbone module - a convolutional (in which case conv_inputs must be supplied) or some other (in which case backbone_args must be supplied) neural network
        :param heads:           Sequence of head modules
        """
        super().__init__()
        if (conv_inputs is not None) and (backbone_args is not None):
            raise Exception('Both conv_inputs and backbone_args are set. Only one may be set (conv_inputs soon to be deprecated)')
        if (conv_inputs is None) and (backbone_args is None):
            raise Exception('Neither conv_inputs nor backbone_args are None. One must be set (conv_inputs soon to be deprecated)')

        self.conv_inputs = conv_inputs
        self.backbone_args = backbone_args
        self.backbone = backbone
        self.add_module('backbone', self.backbone)
        self.heads = torch.nn.ModuleList(heads)
        self.add_module('heads', self.heads)

    def forward(self,
                batch_dict: Dict) -> Dict:
        if self.conv_inputs is not None:
            conv_input = torch.cat([FuseUtilsHierarchicalDict.get(batch_dict, conv_input[0]) for conv_input in self.conv_inputs], 1)
            backbone_features = self.backbone.forward(conv_input)
        else:
            backbone_args = [FuseUtilsHierarchicalDict.get(batch_dict, inp[0]) for inp in self.backbone_args]
            backbone_features = self.backbone.forward(*backbone_args)

        FuseUtilsHierarchicalDict.set(batch_dict, 'model.backbone_features', backbone_features)

        for head in self.heads:
            batch_dict = head.forward(batch_dict)

        return batch_dict['model']


if __name__ == '__main__':
    from fuse.models.heads.head_dense_segmentation import FuseHeadDenseSegmentation
    import torch
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

    DEVICE = 'cpu'  # 'cuda'
    DATAPARALLEL = False  # True

    model = FuseModelDefault(
        conv_inputs=(('data.input.input_0.tensor', 1),),
        backbone=FuseBackboneInceptionResnetV2(),
        heads=[
            FuseHeadGlobalPoolingClassifier(head_name='head_0',
                                            conv_inputs=[('model.backbone_features', 384)],
                                            post_concat_inputs=None,
                                            num_classes=2),

            FuseHeadDenseSegmentation(head_name='head_1',
                                      conv_inputs=[('model.backbone_features', 384)],
                                      num_classes=2)
        ]
    )

    model = model.to(DEVICE)
    if DATAPARALLEL:
        model = torch.nn.DataParallel(model)

    dummy_data = {'data':
                      {'input':
                           {'input_0': {'tensor': torch.zeros([17, 1, 200, 100]).to(DEVICE), 'metadata': None}},
                       'gt':
                           {'gt_0': torch.zeros([0]).to(DEVICE)}
                       }
                  }

    res = {}
    res['model'] = model.forward(dummy_data)
    print('Forward pass shape - head_0: ', end='')
    print(str(res['model']['logits']['head_0'].shape))

    print('\nForward pass shape - head_1: ', end='')
    print(str(res['model']['logits']['head_1'].shape))

    total_params = sum(p.numel() for p in model.parameters())
    if not DATAPARALLEL:
        backbone_params = sum(p.numel() for p in model._modules['backbone'].parameters())
        print('Backbone params = %d' % backbone_params)
        print('Heads params = %d' % (total_params - backbone_params))

    print('\nTotal params = %d' % total_params)
