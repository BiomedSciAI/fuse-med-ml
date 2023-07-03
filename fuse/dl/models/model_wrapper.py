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

from typing import Sequence, Dict, Callable, Union, Any

import torch
import torch.nn as nn
from torch import Tensor

from fuse.utils.ndict import NDict


class ModelWrapSeqToDict(nn.Module):
    """
    Fuse model wrapper for wrapping torch modules and passing through Fuse
    """

    def __init__(
        self,
        *,  # preventing positional args
        model: torch.nn.Module = None,
        model_inputs: Union[Sequence[str], Dict[str, str]] = None,
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
            It can also be a dictionary mapping model forward argument name to key used to extract value from batch_dict.
            For example:
            class MyModel(nn.Module):
                def forward(encoder_input, decoder_input):
                    ...

            model_inputs = dict(encoder_input="data.query.encoder_input", decoder_input="data.query.decoder_input")

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
        if isinstance(self.model_inputs, str) or isinstance(self.model_outputs, str):
            raise Exception(
                "Model Inputs and Outputs should be a Sequence of keys to data in a batch NDict. Not str. See fuse.data for more info."
            )

    def forward(self, batch_dict: NDict, *args: Any, **kwargs: Dict[str, Any]) -> NDict:
        # convert input to the model's expected input
        if isinstance(self.model_inputs, dict):
            model_input = {
                input_arg_name: batch_dict[input_batch_dict_key]
                for input_arg_name, input_batch_dict_key in self.model_inputs.items()
            }
        else:
            model_input = [
                batch_dict[input_batch_dict_key]
                for input_batch_dict_key in self.model_inputs
            ]

        # convert input to model expected input
        if self.pre_forward_processing_function is not None:
            model_input = self.pre_forward_processing_function(model_input)

        # run the model
        if isinstance(self.model_inputs, dict):
            model_output = self.model(*args, **model_input, **kwargs)
        else:
            model_output = self.model(*model_input, *args, **kwargs)

        # convert output of model to Fuse expected output
        if self.post_forward_processing_function is not None:
            model_output = self.post_forward_processing_function(model_output)

        if len(self.model_outputs) == 1:
            batch_dict[self.model_outputs[0]] = model_output
        else:
            for i, output_name in enumerate(self.model_outputs):
                if output_name is None:
                    continue
                batch_dict[output_name] = model_output[i]

        return batch_dict

    def __getattr__(self, name: str) -> Union[Tensor, nn.Module]:
        try:
            return super().__getattr__(name)
        except:
            return self.model.__getattribute__(name)


class ModelWrapDictToSeq(nn.Module):
    """
    Fuse model wrapper for wrapping fuse pytorch model and make him be in basic format- input is tensor and output is tensor
    The user need to provide the input and output keys of the fuse model
    """

    def __init__(self, fuse_model: nn.Module, output_key: str, input_key: str):
        super().__init__()
        self.model = fuse_model
        self.output_key = output_key
        self.input_key = input_key

    def forward(self, input: Tensor) -> Tensor:
        batch_dict = NDict()
        # find input key
        batch_dict[self.input_key] = input
        # feed fuse model with dict as he expect
        ans_ndict = self.model(batch_dict)
        # extract model output from dict
        output = ans_ndict[self.output_key]
        return output
