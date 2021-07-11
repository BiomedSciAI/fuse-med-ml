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

import logging
import traceback
from typing import Dict, Optional, Callable, List

import numpy as np
import sklearn.metrics

from fuse.metrics.metric_base import FuseMetricBase
from fuse.metrics.metrics_toolbox import FuseMetricsToolBox


class FuseMetricROCCurve(FuseMetricBase):
    """
    Output the roc curve (Multi class version - one vs rest)
    """

    def __init__(self,
                 pred_name: str,
                 target_name: str,
                 output_filename: str,
                 class_names: Optional[List[str]] = None,
                 filter_func: Optional[Callable] = None,
                 use_sample_weights: Optional[bool] = False,
                 sample_weight_name: Optional[str] = None) -> None:
        """
        :param pred_name:           batch_dict key for predicted output (e.g., class probabilities after softmax)
        :param target_name:         batch_dict key for target (e.g., ground truth label)
        :param output_filename:     output filename to save the figure
        :param class_names:         Optional - name for each class otherwise the class index will be used.
        :param filter_func:         Optional - function that filters batch_dict. The function gets as input batch_dict and returns filtered batch_dict.
                                    When None (default), the entire batch dict is collected and processed.
        :param use_sample_weights:  Optional - when True (default is False), metrics are computed with sample weigths.
                                    For this, use argument 'sample_weight_name' to define the batch_dict key for the sample weight
        :param sample_weight_name:  Optional, when use_sample_weights is True, user must specify the key for collecting sample weights
        """
        super().__init__(pred_name, target_name, filter_func=filter_func,
                         use_sample_weights=use_sample_weights, sample_weight_name=sample_weight_name)
        self._class_names = class_names
        self._output_filename = output_filename

    def process(self) -> Dict[str, float]:
        """
        Save to a file ROC curve and AUC values.
        :return: an empty dictionary
        """
        results = {}
        num_classes = self.collected_data['pred'][0].shape[0]
        class_names = FuseMetricsToolBox.get_class_names(num_classes, self._class_names)

        sample_weight = self.collected_data['sample_weight'] if self.use_sample_weights else None

        try:
            FuseMetricsToolBox.roc_curve(self.epoch_preds, self.epoch_targets, sample_weight, class_names, self._output_filename)
        except:
            track = traceback.format_exc()
            logging.getLogger('Fuse').error(track)

        return results

