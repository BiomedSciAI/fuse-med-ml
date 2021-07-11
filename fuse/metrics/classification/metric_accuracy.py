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

from typing import Optional, Callable, List

import numpy as np

from fuse.metrics.metric_base import FuseMetricBase


class FuseMetricAccuracy(FuseMetricBase):
    """
    Basic Accuracy - Operation point is not used -
    the predicted class will simply be the one with the max probability
    """

    def __init__(self,
                 pred_name: str,
                 target_name: str,
                 class_names: Optional[List[str]] = None,
                 filter_func: Optional[Callable] = None,
                 use_sample_weights: Optional[bool] = False,
                 sample_weight_name: Optional[str] = None) -> None:
        """
        :param pred_name:           batch_dict key for predicted output (e.g., class probabilities after softmax)
        :param target_name:         batch_dict key for target (e.g., ground truth label)
        :param class_names:         Optional - name for each class otherwise the class index will be used.
        :param filter_func:         Optional - function that filters batch_dict. The function gets as input batch_dict and returns filtered batch_dict.
                                    When None (default), the entire batch dict is collected and processed.
        :param use_sample_weights:  Optional - when True (default is False), metrics are computed with sample weights.
                                    For this, use argument 'sample_weight_name' to define the batch_dict key for the sample weight
        :param sample_weight_name:  Optional, when use_sample_weights is True, user must specify the key for collecting sample weights
        """
        super().__init__(pred_name, target_name, filter_func=filter_func,
                         use_sample_weights=use_sample_weights, sample_weight_name=sample_weight_name)
        self._class_names = class_names

    def process(self) -> float:
        """
        :return: accuracy
        """
        if len(self.collected_data['pred']) == 0:
            return 'N/A'

        accurate = 0

        sample_weights = self.collected_data['sample_weight'] if self.use_sample_weights else np.ones(len(self.collected_data['target']))
        weight_sum = 0
        for target, pred, weight in zip(self.collected_data['target'], self.collected_data['pred'], sample_weights):
            pred = np.argmax(pred)
            if target == pred:
                accurate += weight
            weight_sum += weight

        accuracy = float(accurate) / weight_sum
        return accuracy


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

    metric = FuseMetricAccuracy(pred_name='preds', target_name='targets', use_sample_weights=True,
                           sample_weight_name='weights')
    metric.collect(data)
    print("use_sample_weights = True")
    res = metric.process()
    print(res)
    metric.reset()
    metric.collect(data)
    res = metric.process()
    print("use_sample_weights = True (after reset)")
    print(res)

    metric = FuseMetricAccuracy(pred_name='preds', target_name='targets', use_sample_weights=False,
                           sample_weight_name='weights')
    metric.collect(data)
    res = metric.process()
    print("use_sample_weights = False")
    print(res)