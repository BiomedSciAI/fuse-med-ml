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
from torch import Tensor
from typing import List


class BinaryDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-5, reduction: str = "mean"):
        """
        :param eps:         A float number to smooth loss, and avoid NaN error, default: 1
        :param reduction:   Reduction method to apply, return mean over batch if 'mean',
                            return sum if 'sum', return a tensor of shape [N,] if 'none'

        Returns:            Loss tensor according to arg reduction
        Raise:              Exception if unexpected reduction
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def __call__(self, predict: Tensor, target: Tensor) -> Tensor:
        assert (
            predict.shape[0] == target.shape[0]
        ), "predict & target batch size don't match"
        reduce_axis: List[int] = torch.arange(2, len(predict.shape)).tolist()
        intersection = torch.sum(target * predict, dim=reduce_axis)
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(predict, dim=reduce_axis)

        num = 2 * intersection + self.eps
        den = ground_o + pred_o + self.eps
        loss = 1 - num / den

        # return loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))
