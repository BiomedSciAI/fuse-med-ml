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

from typing import Dict, List, Tuple, Callable, Optional, Sequence, Union
import numpy as np

from fuse.metrics.metric_base import FuseMetricBase
from fuse.metrics.metrics_toolbox import FuseMetricsToolBox


class FuseMetricMultiClassBSS(FuseMetricBase):
    """
    Multi class version for Brier Skill Score (BSS)

    """

    def __init__(self, pred_name: str, target_name: str, filter_func: Optional[Callable] = None, **kwargs):
        """
        :param pred_name:       batch_dict key for predicted output (e.g., class probabilities after softmax)
        :param target_name:     batch_dict key for target (e.g., ground truth label)
        :param filter_func: function that filters batch_dict/ The function gets ans input batch_dict and returns filtered batch_dict
        """
        super().__init__(pred_name, target_name, filter_func, **kwargs)

    def process(self) -> float:
        """
        Calculates BSS
        :return: BSS
        """
        # if no samples, return 'N/A'
        if len(self.epoch_preds) == 0:
            return "N/A"

        return multi_class_bss(
            np.array(self.epoch_preds), np.array(self.epoch_targets, dtype=np.int)
        )


def multi_class_bs(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Brier Score:
    bs = 1/N * SUM_n SUM_c (predictions_{n,c} - targets_{n,c})^2
    :param predictions: probability score. Expected Shape [N, C]
    :param targets: target class (int) per sample. Expected Shape [N]
    """
    # create one hot vector
    targets_one_hot = np.zeros_like(predictions)
    targets_one_hot[np.arange(targets_one_hot.shape[0]), targets] = 1

    return float(np.mean(np.sum((predictions - targets_one_hot) ** 2, axis=1)))


def multi_class_bss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Brier Skill Score:
    bss = 1 - bs / bs_{ref}

    bs_{ref} will be computed for a model that makes a predictions according to the prevalance of each class in dataset

    :param predictions: probability score. Expected Shape [N, C]
    :param targets: target class (int) per sample. Expected Shape [N]
    """

    # BS
    bs = multi_class_bs(predictions, targets)

    # no skill BS
    no_skill_prediction = [(targets == target_cls).sum() / targets.shape[0] for target_cls in
                           range(predictions.shape[-1])]
    no_skill_predictions = np.tile(np.array(no_skill_prediction), (predictions.shape[0], 1))
    bs_ref = multi_class_bs(no_skill_predictions, targets)

    return 1.0 - bs / bs_ref
