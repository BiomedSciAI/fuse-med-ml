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

from typing import Sequence, Dict, Tuple, Callable

import torch

from fuse.models.backbones.backbone_inception_resnet_v2 import FuseBackboneInceptionResnetV2
from fuse.models.heads.head_global_pooling_classifier import FuseHeadGlobalPoolingClassifier
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseModelWrapper(torch.nn.Module):
    """
    Fuse model wrapper for wrapping torch modules and passing through Fuse
    """

    def __init__(self,
                 model: torch.nn.Module = None,
                 model_inputs: Sequence[str] = ('data.input.input_0.tensor',),
                 pre_forward_processing_function: Callable = None,
                 post_forward_processing_function: Callable = None,
                 model_outputs: Sequence[str] = ('output.output_0',)
                 ) -> None:
        """
        Default Fuse model wrapper - extracts batch_dict data from model_inputs keys and calls forward of model.
        Then, puts forward's output into batch_dict model_output keys.
        For pre processing of the batch dict data use pre_forward_processing_function.
        For post processing of the forward's output data use post_forward_processing_function.

        :param model: The model to wrap
        :param model_inputs: sequence of keys in batch dict to transfer into model.forward function
        :param pre_forward_processing_function: utility function to process input data before forward is called (after it's extracted from batch_dict)
        :param post_forward_processing_function: utility function to process forward's output data before it is saved into batch_dict
        :param model_outputs: keys in batch dict to save the outputs of model.forward()

        """
        super().__init__()
        self.model_inputs = model_inputs
        self.model = model
        self.add_module('wrapped_model', self.model)
        self.pre_forward_processing_function = pre_forward_processing_function
        self.post_forward_processing_function = post_forward_processing_function
        self.model_outputs = model_outputs

    def forward(self,
                batch_dict: Dict) -> Dict:
        # convert input to the model's expected input
        model_input = [FuseUtilsHierarchicalDict.get(batch_dict, conv_input) for conv_input in self.model_inputs]

        # convert input to model expected input
        if self.pre_forward_processing_function is not None:
            model_input = self.pre_forward_processing_function(model_input)

        # run the model
        model_output = self.model.forward(*model_input)

        # convert output of model to Fuse expected output
        if self.post_forward_processing_function is not None:
            model_output = self.post_forward_processing_function(model_output)

        if len(self.model_outputs) == 1:
            FuseUtilsHierarchicalDict.set(batch_dict, 'model.' + self.model_outputs[0], model_output)
        else:
            for i, output_name in enumerate(self.model_outputs):
                FuseUtilsHierarchicalDict.set(batch_dict, 'model.' + output_name, model_output[i])

        return batch_dict['model']


if __name__ == '__main__':
    from torchvision.models.googlenet import BasicConv2d
    import torchvision.models as models
    import torch
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

    DEVICE = 'cpu'  # 'cuda'
    DATAPARALLEL = False  # True

    googlemet_model = models.googlenet.GoogLeNet()
    googlemet_model.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3)

    def convert_googlenet_outputs(output):
        return output.logits

    model = FuseModelWrapper(
        model_inputs=['data.input.input_0.tensor'],
        model=googlemet_model,
        post_forward_processing_function=convert_googlenet_outputs,
        model_outputs=['google.output.logits']
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
    print('Forward pass shape: ', end='')
    print('logits', str(res['model']['google']['output']['logits'].shape))

