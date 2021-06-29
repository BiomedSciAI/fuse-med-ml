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

import numpy as np
import sklearn.metrics
from typing import Dict, Optional, Callable, Union, List, Tuple

from Fuse.metrics.metric_base import FuseMetricBase
from Fuse.metrics.metrics_toolbox import FuseMetricsToolBox


class FuseMetricConfusionMatrix(FuseMetricBase):
    """
    Computes the confusion matrix from the data
    Multi class version
    Will add to epoch results:
    metrics.<metric_name>.<cls> - array of predicted labels for each class index out of the true labels
        e.g.,
            {metrics.<metric_name>.'class_0': array([2, 2, 1]),
            metrics.<metric_name>.'class_1': array([0, 1, 2]),
            metrics.<metric_name>.'class_2': array([1, 0, 1])}
        here, out of 5 known class_0 labels: 2 are true prediction, 2 were predicted to class_1, 1 to class_1
    """

    def __init__(self, pred_name: str,
                 target_name: str,
                 class_names: Optional[List[str]] = None,
                 operation_point:Optional[Union[float, List[Tuple], Callable]] = None,
                 filter_func: Optional[Callable] = None,
                 use_sample_weights: Optional[bool] = False,
                 sample_weight_name: Optional[str] = None) -> None:
        """
        :param pred_name:       batch_dict key for predicted output (e.g., class probabilities after softmax)
        :param target_name:     batch_dict key for target (e.g., ground truth label)
        :param class_names:     Optional - name for each class otherwise the class index will be used.
        :param operation_point: Optional - few possible options:
                                    (1) None - the predicted class will be selected using argmax
                                    (2) float - binary classification, single threshold for the positive class/
                                    (3) threshold for each class, the order of the list is the order in which class
                                    is assigned, Each element in the list is a tuple of (class_idx, class_threshold)
                                    (4) Callable - getting as an input the prediction and targets and returns the format specified in (3)
        :param filter_func:     Optional - function that filters batch_dict. The function gets as input batch_dict and returns filtered batch_dict.
                                When None (default), the entire batch dict is collected and processed.
        :param use_sample_weights:  Optional - when True (default is False), metrics are computed with sample weigths.
                                    For this, use argument 'sample_weight_name' to define the batch_dict key for the sample weight
        :param sample_weight_name:  Optional, when use_sample_weights is True, user must specify the key for collecting sample weights
        """
        super().__init__(pred_name, target_name, filter_func=filter_func,
                         use_sample_weights=use_sample_weights, sample_weight_name=sample_weight_name)
        self._class_names = class_names
        self._operation_point = operation_point


    def process(self) -> Dict[str, float]:
        """
        Calculates Confusion Matrix per epoch
        :return: dictionary including Area under ROC curve (floating point in range [0, 1]/-1) for each class (class vs rest),
                 -1 will be set for invalid/undefined result (an error message will be printed)
                 The dictionary will also include the average AUC
        """
        # find operation points
        if isinstance(self._operation_point, Callable):
            class_thresholds = self._operation_point(self.epoch_preds, self.epoch_targets)
        elif isinstance(self._operation_point, float):
            class_thresholds = [(1, self._operation_point), (0, 0.0)]
        else:
            class_thresholds = self._operation_point

        results = {}
        # convert probabilities to class predictions
        epoch_class_preds = FuseMetricsToolBox.convert_probabilities_to_class(self.epoch_preds, class_thresholds)

        # set class names if not provided by the user
        class_names = FuseMetricsToolBox.get_class_names(self.epoch_preds[0].shape[-1], self._class_names)

        sample_weights = self.collected_data['sample_weight'] if self.use_sample_weights else None

        conf_matrix = sklearn.metrics.confusion_matrix(self.collected_data['target'], epoch_class_preds, sample_weight=sample_weights)
        for cls_ind, cls in enumerate(class_names):
            results[cls] = conf_matrix[cls_ind]

        return results


if __name__ == '__main__':
    data = {'preds': np.array([[0.8, 0.1, 0.1],
                               [0.5, 0.3, 0.2],
                               [0.6, 0.3, 0.1],
                               [0.6, 0.1, 0.3],
                               [0.7, 0.2, 0.1],
                               [0.3, 0.2, 0.5],
                               [0.1, 0.2, 0.7],
                               [0.2, 0.6, 0.2],
                               [0.3, 0.3, 0.4],
                               [0.7, 0.2, 0.1]]),
            'targets': np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2]),
            'weights': np.array([0.03, 0.9, 0.05, 0.52, 0.23, 0.72, 0.13, 0.113, 0.84, 0.09])}

    print("threshold=((0, 0.6), (1, 0.5), (2, 0.2))")
    metric = FuseMetricConfusionMatrix(pred_name='preds', target_name='targets',
                                       operation_point=[(0, 0.6), (1, 0.5), (2, 0.2)], use_sample_weights=True,
                                       sample_weight_name='weights')
    metric.collect(data)
    res = metric.process()
    print('with weights')
    for k, v in res.items():
        print(k, '\t', v)
    metric = FuseMetricConfusionMatrix(pred_name='preds', target_name='targets',
                                       operation_point=[(0, 0.6), (1, 0.5), (2, 0.2)], use_sample_weights=False,
                                       sample_weight_name='weights')
    metric.collect(data)
    res = metric.process()
    print('no weights')
    for k, v in res.items():
        print(k, '\t', v)

    print('no threshold (argmax)')
    metric = FuseMetricConfusionMatrix(pred_name='preds', target_name='targets')
    metric.collect(data)
    res = metric.process()
    for k, v in res.items():
        print(k, '\t', v)

    data = {'preds': np.array([[0.8, 0.2],
                               [0.5, 0.5],
                               [0.6, 0.4],
                               [0.6, 0.4],
                               [0.7, 0.3],
                               [0.3, 0.7],
                               [0.1, 0.9],
                               [0.2, 0.8],
                               [0.3, 0.7],
                               [0.4, 0.6]]),
            'targets': np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0])}
    metric = FuseMetricConfusionMatrix(pred_name='preds', target_name='targets')
    metric.collect(data)
    res = metric.process()
    print('binary')
    for k, v in res.items():
        print(k, '\t', v)
