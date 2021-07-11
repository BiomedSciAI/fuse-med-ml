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


class FuseMetricConfusion(FuseMetricBase):
    """
    Multi class version for confusion metrics including: sensitivity, specificity, recall., precision, f1.

    """
    def __init__(self, pred_name: str, target_name: str, filter_func: Optional[Callable] = None, class_names:List = None,
                 operation_point:Optional[Union[float, List[Tuple], Callable]] = None,
                 metrics:Sequence[str] = ('sensitivity',), sum_weights:List = None, **kwargs):
        """
        :param pred_name:       batch_dict key for predicted output (e.g., class probabilities after softmax)
        :param target_name:     batch_dict key for target (e.g., ground truth label)
        :param filter_func: function that filters batch_dict/ The function gets ans input batch_dict and returns filtered batch_dict
        :param class_names:     Optional - name for each class otherwise the class index will be used.
        :param operation_point: Optional - few possible options:
                                    (1) None - the predicted class will be selected using argmax
                                    (2) float - binary classification, single threshold for the positive class/
                                    (3) threshold for each class, the order of the list is the order in which class
                                    is assigned, Each element in the list is a tuple of (class_idx, class_threshold)
                                    (4) Callable - getting as an input the prediction and targets and returns the format specified in (3)
        :param metrics: list of metrics name to return. See FuseMetricsToolBox.confusion_metrics for available options.
        :param sum_weights: weights for the weighted sum of sensitivity average

        """
        super().__init__(pred_name, target_name, filter_func, **kwargs)
        self._class_names = class_names
        self._operation_point = operation_point
        self._metrics = metrics
        self._sum_weights = sum_weights

    def process(self) -> Dict[str, float]:
        """
        Calculates the confusion metrics
        :return: dictionary including per class in one vs all manner and a macro avg for each class included in metrics

        """
        # find operation points
        if isinstance(self._operation_point, Callable):
            class_thresholds = self._operation_point(self.epoch_preds, self.epoch_targets)
        elif isinstance(self._operation_point, float):
            class_thresholds = [(1, self._operation_point), (0, 0.0)]
        else:
            class_thresholds = self._operation_point

        agg_results = {}

        # if no samples, return 'N/A'
        if len(self.epoch_preds) == 0:
            for metric_name in self._metrics:
                agg_results[metric_name + '_macro_avg'] = 'N/A'
            return agg_results

        # convert probabilities to class predictions
        epoch_class_preds = FuseMetricsToolBox.convert_probabilities_to_class(self.epoch_preds, class_thresholds)

        # set class names if not provided by the user
        class_names = FuseMetricsToolBox.get_class_names(self.epoch_preds[0].shape[-1], self._class_names)

        # set weight per class if not provided by the user
        if self._sum_weights is None:
            self._sum_weights = [1] * len(class_names)
        self._sum_weights = np.array(self._sum_weights)

        # compute the confusion metrics
        for cls_ind, cls in enumerate(class_names):
            cls_result = FuseMetricsToolBox.confusion_metrics(epoch_class_preds, cls_ind, np.array(self.epoch_targets), self._metrics)
            for metric_name in cls_result:
                if metric_name not in agg_results:
                    agg_results[metric_name] = {}
                metric_result = agg_results[metric_name]
                metric_result[cls] = cls_result[metric_name]

        # pack it in a dictionary and calculate macro average
        results = {}
        for metric_name in self._metrics:
            value_per_class = np.array(list(agg_results[metric_name].values()))
            indices = ~np.isnan(value_per_class)
            results.update({metric_name + '_' + cls: agg_results[metric_name][cls] for cls in agg_results[metric_name]})
            results[metric_name + '_macro_avg'] = np.average(value_per_class[indices], weights=self._sum_weights[indices])

        return results