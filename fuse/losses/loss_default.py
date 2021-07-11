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

from typing import Callable, Dict, Optional

import torch

from fuse.losses.loss_base import FuseLossBase
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseLossDefault(FuseLossBase):
    """
    Default Fuse loss function
    """

    def __init__(self,
                 pred_name: str = None,
                 target_name: str = None,
                 batch_kwargs_name: str = None,
                 callable: Callable = None,
                 sample_weight_name: Optional[str] = None,
                 weight: float = 1.0,
                 filter_func: Optional[Callable] = None,
                 **kwargs
                 ) -> None:
        """
        This class wraps a PyTorch loss function with a Fuse api.
        :param pred_name:               batch_dict key for prediction (e.g., network output)
        :param target_name:             batch_dict key for target (e.g., ground truth label)
        :param batch_kwargs_name:       batch_dict key for additional, ad-hoc kwargs for loss function
                                        Note: batch_kwargs will be merged into other loss function kwargs
        :param sample_weight_name       batch_dict key that holds the sample weight for loss summation
        :param callable:                PyTorch loss function handle (e.g., torch.nn.functional.cross_entropy)

        :param weight:                  Weight multiplier for final loss value
        :param filter_func:             function that filters batch_dict/ The function gets ans input batch_dict and returns filtered batch_dict
        :param kwargs:                  kwargs for PyTorch loss function
        """
        super().__init__()
        self.pred_name = pred_name
        self.target_name = target_name
        self.batch_kwargs_name = batch_kwargs_name
        self.callable = callable
        self.sample_weight_name = sample_weight_name
        self.weight = weight
        self.filter_func = filter_func
        self.kwargs = kwargs

    def __call__(self, batch_dict: Dict) -> torch.Tensor:
        # filter batch_dict if required
        if self.filter_func is not None:
            batch_dict = self.filter_func(batch_dict)
        preds = FuseUtilsHierarchicalDict.get(batch_dict, self.pred_name)
        targets = FuseUtilsHierarchicalDict.get(batch_dict, self.target_name)
        batch_kwargs = FuseUtilsHierarchicalDict.get(batch_dict, self.batch_kwargs_name) if self.batch_kwargs_name is not None else {}
        kwargs_copy = self.kwargs.copy()
        kwargs_copy.update(batch_kwargs)
        if self.sample_weight_name is not None:
            assert 'reduction' not in kwargs_copy.keys(), 'reduction is forced to none when applying sample weight'
            kwargs_copy.update({'reduction': 'none'})
        loss_obj = self.callable(preds, targets, **kwargs_copy) * self.weight
        if self.sample_weight_name is not None:
            sample_weight = FuseUtilsHierarchicalDict.get(batch_dict, self.sample_weight_name)
            weighted_loss = loss_obj*sample_weight
            loss_obj = torch.mean(weighted_loss)

        return loss_obj


if __name__ == '__main__':
    import torch

    batch_dict = {'pred': torch.randn(3, 5, requires_grad=True),
                  'gt': torch.empty(3, dtype=torch.long).random_(5),
                  'batch_loss_kwargs': {'reduction': 'mean', 'ignore_index': 0}}

    loss = FuseLossDefault(pred_name='pred',
                           target_name='gt',
                           batch_kwargs_name='batch_loss_kwargs',
                           callable=torch.nn.functional.cross_entropy,
                           weight=1.0,
                           reduction='sum')

    res = loss(batch_dict)
    print('Loss output = ' + str(res))
