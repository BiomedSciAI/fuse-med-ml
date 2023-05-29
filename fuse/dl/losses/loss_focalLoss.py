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
from torch import Tensor
import numpy as np


def make_one_hot(input: Tensor, num_classes: int) -> Tensor:
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
    result = torch.zeros(shape, device=input.device)
    result = result.scatter_(1, input, 1)

    return result


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        FL = -alpha(1-pt)^gamma * log(pt)
        :param alpha: hyperparameter weighted term
        :param gamma: hyperparameter
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:

        if targets.dim() < 4:
            targets = targets.unsqueeze(1)
        targets = make_one_hot(targets, inputs.shape[1])

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss).data.view(-1)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss.data.view(-1)
        return F_loss.mean()
