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

from fuse.metrics.metrics_toolbox import FuseMetricsToolBox

from fuse.metrics.metric_base import FuseMetricBase
from typing import Dict, List, Callable, Optional, Sequence
import numpy as np


class FuseMetricConfusionIndeterminate(FuseMetricBase):
    """
    Multi class version for confusion metrics such as sensitivity,specificity, etc... Given an option to filter out indeterminate predictions.

    """

    def __init__(self, pred_name: str, target_name: str, filter_func: Optional[Callable] = None, class_names: List = None,
                 indeterminate_thresholds: Sequence[float] = (0.5,), indeterminate_classes: Sequence[int] = tuple(),
                 determinate_percent_mode: bool = False,
                 metrics: Sequence[str] = ('sensitivity',), sum_weights: List = None, **kwargs):
        """
        :param pred_name:       batch_dict key for predicted output (e.g., class probabilities after softmax)
        :param target_name:     batch_dict key for target (e.g., ground truth label)
        :param class_names:     Optional - name for each class otherwise the class index will be used.
        :param filter_func: function that filters batch_dict The function gets ans input batch_dict and returns filtered batch_dict
        :param indeterminate_thresholds: threshold for indeterminate class or if determinate_percent_mode set to True: target percent of samples to predict
                                        (the rest will be marked as indeterminate)
        :param indeterminate_classes: if the sample prediction included in indeterminate_classes, it will be marked as indeterminate.
        :param determinate_percent_mode: see indeterminate_thresholds
        :param metrics: list of metrics name to return. See FuseMetricsToolBox.confusion_metrics for available options.
        :param sum_weights: weights for the weighted sum of sensitivity average

        """
        super().__init__(pred_name, target_name, filter_func, **kwargs)
        self._class_names = class_names
        self._indeterminate_thresholds = indeterminate_thresholds
        self._indeterminate_classes = indeterminate_classes
        self._determinate_percent_mode = determinate_percent_mode
        self._metrics = metrics
        self._sum_weights = sum_weights

    def process(self) -> Dict:
        """
        Calculates sensitivity
        :return: dictionary including sensitivity for each class (class vs rest) given indeterminate threshold,
        Expected output per indeterminate threshold:
        '
        ind_th_<indeterminate threshold>.determinate_percentage: <value>
        ind_th_<indeterminate threshold>.<metric>_<class name 0>: <value>
        ind_th_<indeterminate threshold>.<metric>_<class name 1>_sens: <value>
        ...
        ind_th_<indeterminate threshold>.<metric>_macro_avg: <value>
        '

        """
        results = {}

        # set class names if not provided by the user
        class_names = FuseMetricsToolBox.get_class_names(self.epoch_preds[0].shape[-1], self._class_names)

        # set weight per class if not provided by the user
        if self._sum_weights is None:
            self._sum_weights = [1] * len(class_names)
        self._sum_weights = np.array(self._sum_weights)

        epoch_preds = np.array(self.epoch_preds)
        epoch_targets = np.array(self.epoch_targets)
        epoch_class_preds = epoch_preds.argmax(axis=1)
        epoch_score_preds = epoch_preds.max(axis=1)
        epoch_score_preds_lst = [epoch_score_preds[epoch_class_preds == i] for i in range(len(self._class_names))]

        for threshold in self._indeterminate_thresholds:

            # convert percentile to score threshold
            if self._determinate_percent_mode:
                if isinstance(threshold, list):
                    threshold = [np.percentile(preds, (1.0 - threshold[i]) * 100.0) for i, preds in enumerate(epoch_score_preds_lst)]
                else:
                    threshold = np.percentile(epoch_score_preds, (1.0 - threshold) * 100.0)

            # mark prediction lower than the threshold as indeterminate
            if isinstance(threshold, list):
                threshold = np.array(threshold)
                determinate_indices = (epoch_score_preds >= threshold[epoch_class_preds])
            else:
                determinate_indices = epoch_score_preds >= threshold

            # mark indeterminate classes as in indeterminate
            for cls_ind in self._indeterminate_classes:
                determinate_indices = determinate_indices & (epoch_class_preds != cls_ind)

            # filter indeterminate samples
            results[f'ind_th_{threshold}_determinate_percentage'] = np.round(determinate_indices.sum() / epoch_preds.shape[0], 3)
            preds = epoch_class_preds[determinate_indices]
            targets = epoch_targets[determinate_indices]

            if preds.shape[0] == 0:
                for metric_name in self._metrics:
                    results[f'ind_th_{threshold}_{metric_name}_macro_avg'] = 'N/A'

            # compute the confusion metrics
            agg_results = {}
            for cls_ind, cls in enumerate(class_names):
                cls_result = FuseMetricsToolBox.confusion_metrics(preds, cls_ind, targets, self._metrics)

                results[f'ind_th_{threshold}_determinate_percentage_{cls}'] = np.round(
                    np.where(targets == cls_ind, 1, 0).sum() / np.where(epoch_targets == cls_ind, 1, 0).sum(), 3)

                for metric_name in self._metrics:
                    results[f'ind_th_{threshold}_{metric_name}_{cls}'] = cls_result[metric_name]

                # aggregate results
                for metric_name in self._metrics:
                    if metric_name not in agg_results:
                        agg_results[metric_name] = {}
                    metric_result = agg_results[metric_name]
                    metric_result[cls] = cls_result[metric_name]

            # calculate macro average
            for metric_name in self._metrics:
                value_per_class = np.array(list(agg_results[metric_name].values()))
                indices = ~np.isnan(value_per_class)
                results[f'ind_th_{threshold}_{metric_name}_macro_avg'] = np.average(value_per_class[indices], weights=self._sum_weights[indices])

        return results
