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

from typing import Callable, Optional

import torch

from fuse.dl.losses.loss_base import LossBase
from fuse.utils.ndict import NDict


class LossDefault(LossBase):
    """
    Default Fuse loss function

    Basic Usage Example:
    '''
    from fuse.dl.losses.loss_default import LossDefault

    my_loss_func = LossDefault(
        pred='model.preds',
        target='data.groundtruth.targets',
        callable=torch.nn.functional.cross_entropy,
        )

    batch_dict = # batch_dict with pred and target
    loss = my_loss_func(batch_dict)
    '''


    Custom Preprocessing Example:
    Sometimes you want custom preprocessing of the batch_dict - for example in the following scenario:

    A multi-head / multi-task model, in which you have ground truth labels only for a subset of the samples.
    In such case, you may use the optional "preprocess_func" to filter out the samples that you don't have labels for both tasks.

    Example code:
    '''
    def my_preprocess_func(batch_dict: NDict) -> NDict:
        # filter out samples that don't have labels for both tasks
        keep_indices = #... calculate which indices to keep
        batch_dict = batch_dict.indices(keep_indices) #this will keep only a subset of elements per each key in the dictionary (where applicable)
        return batch_dict

    my_loss_func = LossDefault(
        pred='model.preds',
        target='data.groundtruth.targets',
        callable=torch.nn.functional.cross_entropy,
        preprocess_func=my_preprocess_func,
        )
    '''

    """

    def __init__(
        self,
        *,  # prevent positional args
        pred: str = None,
        target: str = None,
        callable: Callable = None,
        weight: Optional[float] = None,
        preprocess_func: Optional[Callable] = None,
    ) -> None:
        """
        This class wraps a PyTorch loss function with a Fuse api.
        Args:
        :param pred:               batch_dict key for prediction (e.g., network output)
        :param target:             batch_dict key for target (e.g., ground truth label)
        :param callable:           PyTorch loss function handle (e.g., torch.nn.functional.cross_entropy)
        :param weight:             scalar loss multiplier
        :param preprocess_func:             function that filters batch_dict/ The function gets an input batch_dict and returns filtered batch_dict
            the expected function signature is:
                foo(batch_dict: NDict) -> NDict:
        """
        super().__init__()
        self.pred = pred
        self.target = target
        self.callable = callable
        self.weight = weight
        self.preprocess_func = preprocess_func

    def forward(self, batch_dict: NDict) -> torch.Tensor:
        # preprocess batch_dict if required
        if self.preprocess_func is not None:
            batch_dict = self.preprocess_func(batch_dict)
        preds = batch_dict[self.pred]
        targets = batch_dict[self.target]

        loss_obj = self.callable(preds, targets)

        if self.weight is not None:
            loss_obj *= self.weight

        return loss_obj
