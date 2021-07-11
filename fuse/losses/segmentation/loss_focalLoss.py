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

from typing import Callable, Dict, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fuse.losses.loss_base import FuseLossBase
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
import numpy as np


def make_one_hot(input, num_classes, device='cuda'):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape, device=device)
    result = result.scatter_(1, input, 1)

    return result


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha: float =.25, gamma: float =2.):
        '''
        FL = -alpha(1-pt)^gamma * log(pt)
        :param alpha: hyperparameter weighted term
        :param gamma: hyperparameter
        '''
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):

        if targets.dim() < 4:
            targets = targets.unsqueeze(1)
        targets = make_one_hot(targets, inputs.shape[1])

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss).data.view(-1)
        F_loss = at*(1-pt)**self.gamma * BCE_loss.data.view(-1)
        return F_loss.mean()


class FuseFocalLoss(FuseLossBase):

    def __init__(self,
                 pred_name: str = None,
                 target_name: str = None,
                 weight: float = 1.0,
                 alpha: float=.25,
                 gamma: float=2,
                 filter_func: Optional[Callable] = None,
                 resize_mode: str = 'interpolate') -> None:
        """
        :param pred_name:               batch_dict key for predicted logits (e.g., class probabilities BEFORE softmax).
                                        Expected Tensor shape = [batch, num_classes, height, width]
        :param target_name:             batch_dict key for target (e.g., ground truth label). Expected Tensor shape = [batch, height, width]
        :param callable:                PyTorch loss function handle (e.g., torch.nn.functional.cross_entropy)
        :param alpha:                   hyperparameter for focal loss function
        :param gamma:                   hyperparameter for focal loss function
        :param weight:                  Weight multiplier for final loss value
        :param filter_func:             function that filters batch_dict/ The function gets ans input batch_dict and returns filtered batch_dict
        :param resize_mode:             Resize mode- either using a max pooling kernel(default), or using PyTorch interpolation ('interpolate'/'maxpool')
        """
        super().__init__(pred_name, target_name, weight)
        self.class_weights = class_weights
        self.filter_func = filter_func
        self.resize_mode = resize_mode
        self.callable = WeightedFocalLoss(alpha=alpha, gamma=gamma)

    def __call__(self, batch_dict: Dict) -> torch.Tensor:
        # filter batch_dict if required
        if self.filter_func is not None:
            batch_dict = self.filter_func(batch_dict)

        preds = FuseUtilsHierarchicalDict.get(batch_dict, self.pred_name)  # preds shape [batch_size, num_classes, height, width]
        targets = FuseUtilsHierarchicalDict.get(batch_dict, self.target_name)  # targets shape [batch_size, height, width]

        batch_size, targets_height, targets_width = targets.shape
        batch_size, num_classes, preds_height, preds_width = preds.shape

        # Resize targets (if needed) to match predicted activation map
        # ============================================================
        targets = targets.reshape([batch_size, 1, targets_height, targets_width])

        if (targets_height > preds_height) or (targets_width > preds_width):
            if targets.dtype != torch.float32:
                targets = targets.type(torch.float32).to(targets.device)
            if self.resize_mode == 'maxpool':
                block_height = int(targets_height / preds_height)
                block_width = int(targets_width / preds_width)
                residual_h = int((targets_height - (block_height * preds_height)) / 2)
                residual_w = int((targets_width - (block_width * preds_width)) / 2)

                targets = torch.nn.functional.max_pool2d(targets[:, :, residual_h:targets_height - residual_h, residual_w:targets_width - residual_w],
                                                         kernel_size=(block_height, block_width))
            elif self.resize_mode == 'interpolate':
                targets = torch.nn.functional.interpolate(targets, size=(preds_height, preds_width))
            else:
                raise Exception

        # Reshape targets and change its dtype to int64 for cross entropy input
        # =====================================================================
        targets = targets.reshape([batch_size, preds_height, preds_width])
        if targets.dtype != torch.int64:
            targets = targets.type(torch.int64).to(targets.device)

        loss_obj = self.callable(preds, targets) * self.weight

        return loss_obj

