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

from typing import Sequence, Dict, Callable

import torch

from fuse.utils.ndict import NDict


class ModelWrapSeqToDict(torch.nn.Module):
    """
    Fuse model wrapper for wrapping torch modules and passing through Fuse
    """

    def __init__(
        self,
        *,  # preventing positional args
        model: torch.nn.Module = None,
        model_inputs: Sequence[str] = None,
        model_outputs: Sequence[str] = None,
        pre_forward_processing_function: Callable = None,
        post_forward_processing_function: Callable = None,
    ) -> None:
        """
        Default Fuse model wrapper - extracts batch_dict data from model_inputs keys and calls forward of model.
        Then, puts forward's output into batch_dict model_output keys.
        For pre processing of the batch dict data use pre_forward_processing_function.
        For post processing of the forward's output data use post_forward_processing_function.

        :param model: The model to wrap
        :param model_inputs: sequence of keys in batch dict to transfer into model.forward function
            for example: model_inputs=('data.input.input_0.tensor',)
        :param model_outputs: keys in batch dict to save the outputs of model.forward()
            for example: model_outputs=('output.output_0',)
        :param pre_forward_processing_function: utility function to process input data before forward is called (after it's extracted from batch_dict)
        :param post_forward_processing_function: utility function to process forward's output data before it is saved into batch_dict



        """
        super().__init__()
        self.model_inputs = model_inputs
        self.model = model
        self.pre_forward_processing_function = pre_forward_processing_function
        self.post_forward_processing_function = post_forward_processing_function
        self.model_outputs = model_outputs

    def forward(self, batch_dict: NDict) -> Dict:
        # convert input to the model's expected input
        model_input = [batch_dict[conv_input] for conv_input in self.model_inputs]

        # convert input to model expected input
        if self.pre_forward_processing_function is not None:
            model_input = self.pre_forward_processing_function(model_input)

        # run the model
        model_output = self.model.forward(*model_input)

        # convert output of model to Fuse expected output
        if self.post_forward_processing_function is not None:
            model_output = self.post_forward_processing_function(model_output)

        if len(self.model_outputs) == 1:
            batch_dict[self.model_outputs[0]] = model_output
        else:
            for i, output_name in enumerate(self.model_outputs):
                batch_dict[output_name] = model_output[i]

        return batch_dict


class ModelWrapDictToSeq(torch.nn.Module):
    """
    Fuse model wrapper for wrapping fuse pytorch model and make him be in basic format- input is tensor and output is tensor
    The user need to provide the input and output keys of the fuse model
    """

    def __init__(self, fuse_model: torch.nn.Module, output_key: str, input_key: str):
        super().__init__()
        self.model = fuse_model
        self.output_key = output_key
        self.input_key = input_key

    def forward(self, input: torch.tensor):
        batch_dict = NDict()
        # find input key
        batch_dict[self.input_key] = input
        # feed fuse model with dict as he excpect
        ans_ndict = self.model(batch_dict)
        # extract model output from dict
        output = ans_ndict[self.output_key]
        return output
