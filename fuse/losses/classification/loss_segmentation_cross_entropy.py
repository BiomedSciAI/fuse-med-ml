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

from typing import Dict, Union

import numpy as np
import torch

from fuse.losses.loss_base import FuseLossBase
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseLossSegmentationCrossEntropy(FuseLossBase):
    def __init__(self,
                 pred_name: str = None,
                 target_name: str = None,
                 weight: float = 1.0,
                 class_weights: Union[str, torch.tensor, None] = 'ipw',
                 resize_mode: str = 'interpolate') -> None:
        """
        SegmentationCrossEntropy loss.
        This class calculates cross entropy loss per location ("dense") of a class activation map ("segmentation"),
        or more correctly, logits of a class activation map. Ground truth map should be a float32 dtype, with binary
        entries. It is resized (reduced) to match dimensions of predicted score map.
        Note: The cross entropy function is weighted with inverse proportion of positive pixels.
        :param pred_name:               batch_dict key for prediction (e.g., network output)
        :param target_name:             batch_dict key for target (e.g., ground truth label)
        :param weight:                  Weight multiplier for final loss value
        :param class_weights:           Class weights. Options:
                                            1. None
                                            2. Tensor with class weights
                                            3. 'ipw' = Inverse proportional weighting
        :param resize_mode:             Resize mode- either using a max pooling kernel, or using PyTorch interpolation (default)
        """

        self.pred_name = pred_name
        self.target_name = target_name
        self.weight = weight
        self.class_weights = class_weights
        self.resize_mode = resize_mode
        assert resize_mode in ['maxpool', 'interpolate']

    def __call__(self, batch_dict: Dict) -> torch.Tensor:
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

        # Generate class weights
        # ======================
        if self.class_weights is None:
            segmentation_weights = None
        elif isinstance(self.class_weights, str) and self.class_weights.lower() == 'ipw':
            # =============================================================================
            # NOTE: PyTorch also has an equivalent 'unique' method with counts, but it was introduced in a later version
            # See: https://github.com/pytorch/pytorch/issues/12598
            values, counts = np.unique(targets.cpu(), return_counts=True)
            # =============================================================================
            values = values.astype('uint16')
            segmentation_weights = np.zeros(num_classes, dtype='float32')
            segmentation_weights[values] = 1 / counts
            segmentation_weights = segmentation_weights * (1 / segmentation_weights.max())
            segmentation_weights = torch.FloatTensor(segmentation_weights).to(targets.device)
        elif isinstance(self.class_weights, (torch.tensor, list, tuple)):
            segmentation_weights = torch.FloatTensor(self.class_weights).to(targets.device)

        # Reshape targets and change its dtype to int64 for cross entropy input
        # =====================================================================
        targets = targets.reshape([batch_size, preds_height, preds_width])
        if targets.dtype != torch.int64:
            targets = targets.type(torch.int64).to(targets.device)

        # Calc loss
        # ==========
        loss_obj = torch.nn.functional.cross_entropy(preds, targets, weight=segmentation_weights) * self.weight

        return loss_obj
