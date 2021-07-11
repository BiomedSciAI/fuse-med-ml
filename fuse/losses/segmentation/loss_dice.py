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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fuse.losses.loss_base import FuseLossBase
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
from typing import Callable, Dict, Optional


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


class BinaryDiceLoss(nn.Module):

    def __init__(self, power: int=1, eps: float =1., reduction: str = 'mean'):
        '''
        :param power:       Denominator value: \sum{x^p} + \sum{y^p}, default: 1
        :param eps:         A float number to smooth loss, and avoid NaN error, default: 1
        :param reduction:   Reduction method to apply, return mean over batch if 'mean',
                            return sum if 'sum', return a tensor of shape [N,] if 'none'

        Returns:            Loss tensor according to arg reduction
        Raise:              Exception if unexpected reduction
        '''
        super(BinaryDiceLoss, self).__init__()
        self.p = power
        self.reduction = reduction
        self.eps = eps

    def __call__(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        if target.dtype == torch.int64:
            target = target.type(torch.float32).to(target.device)
        num = 2*torch.sum(torch.mul(predict, target), dim=1) + self.eps
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.eps
        loss = 1 - num / den

        # return loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class FuseDiceLoss(FuseLossBase):

    def __init__(self, pred_name,
                 target_name,
                 filter_func: Optional[Callable] = None,
                 class_weights=None,
                 ignore_cls_index_list=None,
                 resize_mode: str = 'maxpool',
                 **kwargs):
        '''

        :param pred_name:                batch_dict key for predicted output (e.g., class probabilities after softmax).
                                         Expected Tensor shape = [batch, num_classes, height, width]
        :param target_name:              batch_dict key for target (e.g., ground truth label). Expected Tensor shape = [batch, height, width]
        :param filter_func:              function that filters batch_dict/ The function gets ans input batch_dict and returns filtered batch_dict
        :param class_weights:            An array of shape [num_classes,]
        :param ignore_cls_index_list:    class index to ignore (list)
        :param resize_mode:              Resize mode- either using a max pooling kernel(default), or using PyTorch
                                         interpolation ('interpolate'/'maxpool')
        :param kwargs:                   args pass to BinaryDiceLoss
        '''

        super().__init__(pred_name, target_name, class_weights)
        self.class_weights = class_weights
        self.filter_func = filter_func
        self.kwargs = kwargs
        self.ignore_cls_index_list = ignore_cls_index_list
        self.resize_mode = resize_mode
        self.dice = BinaryDiceLoss(**self.kwargs)

    def __call__(self, batch_dict):

        if self.filter_func is not None:
            batch_dict = self.filter_func(batch_dict)
        predict = FuseUtilsHierarchicalDict.get(batch_dict, self.pred_name).float()
        target = FuseUtilsHierarchicalDict.get(batch_dict, self.target_name).long()

        n, c, h, w = predict.shape
        tar_shape = target.shape
        if len(tar_shape) < 4:
            target = target.unsqueeze(1)
        nt, ct, ht, wt = target.shape

        if h != ht or w != wt:  # upsample
            if self.resize_mode == 'maxpool':
                block_height = int(ht / h)
                block_width = int(wt / w)
                residual_h = int((ht - (block_height * h)) / 2)
                residual_w = int((wt - (block_width * w)) / 2)

                target = torch.nn.functional.max_pool2d(target[:, :, residual_h:ht - residual_h, residual_w:wt - residual_w],
                                                         kernel_size=(block_height, block_width))
            elif self.resize_mode == 'interpolate':
                target = torch.nn.functional.interpolate(target, size=(h, w))
            else:
                raise Exception

        total_loss = 0
        n_classes = predict.shape[1]

        # Convert target to one hot encoding
        if n_classes > 1 and target.shape[1] != n_classes:
            target = make_one_hot(target, n_classes)

        assert predict.shape == target.shape, 'predict & target shape do not match'

        total_class_weights = sum(self.class_weights) if self.class_weights is not None else n_classes
        for cls_index in range(n_classes):
            if cls_index not in self.ignore_cls_index_list:
                dice_loss = self.dice(predict[:, cls_index, :, :], target[:, cls_index, :, :])
                if self.class_weights is not None:
                    assert self.class_weights.shape[0] == n_classes, \
                        'Expect weight shape [{}], got[{}]'.format(n_classes, self.class_weights.shape[0])
                    dice_loss *= self.class_weights[cls_index]
                total_loss += dice_loss
        total_loss /= total_class_weights

        return self.weight*total_loss
