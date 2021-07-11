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
from typing import Callable, Optional, Dict

import numpy as np

from fuse.metrics.metric_base import FuseMetricBase


class FuseMetricDefault(FuseMetricBase):
    """
    Applying a given metric function
    """

    def __init__(self,
                 metric_func: Callable,
                 pred_name: str,
                 target_name: str,
                 class_names: Optional[str] = None,
                 filter_func: Optional[Callable] = None,
                 use_sample_weights: Optional[bool] = False,
                 sample_weight_name: Optional[str] = None) -> None:
        """
        :param metric_func:         metric function. Will be applied for each class in a one-vs-rest manner.
                                    Gets as an input y_true (a boolean np.ndarray), y_score and sample_weight.
        :param pred_name:           batch_dict key for predicted output (e.g., class probabilities after softmax)
        :param target_name:         batch_dict key for target (e.g., ground truth label)
        :param class_names:         Optional - name for each class otherwise the class index will be used.
        :param filter_func:         Optional - function that filters batch_dict/ The function gets ans input batch_dict and returns filtered batch_dict
        :param use_sample_weights:  Optional - when True (default is False), metrics are computed with sample weigths.
                                    For this, use argument 'sample_weight_name' to define the batch_dict key for the sample weight
        :param sample_weight_name:  Optional, when use_sample_weights is True, user must specify the key for collecting sample weights
        """
        super().__init__(pred_name, target_name, filter_func=filter_func,
                         use_sample_weights=use_sample_weights, sample_weight_name=sample_weight_name)
        self._class_names = class_names
        self.metric_func = metric_func

    def process(self) -> Dict[str, float]:
        """
        Computes an evaluation metric per epoch and class - using metric_func
        :return: dictionary including the computed metric for each class (one vs rest),
                 -1 will be set for invalid/undefined result (an error message will be printed)
                 The dictionary will also include the (unweighted) average metric across all classes
        """
        results = {}
        aucs_per_class = []
        if self._class_names is None:
            num_classes = self.collected_data['pred'][0].shape[0]
            class_names = ['class_' + str(i) for i in range(num_classes)]
        else:
            class_names = self._class_names
        sample_weight = self.collected_data['sample_weight'] if self.use_sample_weights else None
        try:
            for cls_ind, cls in enumerate(class_names):
                predicted_ind = [pred[cls_ind] for pred in self.collected_data['pred']]

                computed_metric = self.metric_func(np.asarray(self.collected_data['target']) == cls_ind,
                                                   np.asarray(predicted_ind), sample_weight=sample_weight)
                results[cls] = computed_metric
                aucs_per_class.append(computed_metric)

            results['macro_avg'] = np.nanmean(aucs_per_class)
        except:
            track = traceback.format_exc()
            logging.getLogger('Fuse').error(track)
            results = {cls: -1 for cls in class_names}
            results['macro_avg'] = -1
        return results


if __name__ == '__main__':
    from sklearn import metrics

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
            'weights': np.array([0.03, 0.12, 0.05, 0.52, 0.23, 0.72, 0.13, 0.113, 0.84, 0.09])}


    def f1_score(y_true, y_score, sample_weight):
        return metrics.f1_score(y_true=y_true, y_pred=y_score > 0.5, sample_weight=sample_weight)


    metric_dict = {
        "average_precision": FuseMetricDefault(metric_func=metrics.average_precision_score, pred_name='preds',
                                               target_name='targets',
                                               use_sample_weights=True,
                                               sample_weight_name='weights'),

        "f1 score": FuseMetricDefault(metric_func=f1_score, pred_name='preds',
                                      target_name='targets',
                                      use_sample_weights=True,
                                      sample_weight_name='weights'),

    }

    for metric_name, metric in metric_dict.items():
        print("**** metric=", metric_name)
        metric.collect(data)

        print("use_sample_weights = True")
        res = metric.process()
        for k, v in res.items():
            print(k, '\t', v)

        print("use_sample_weights = False")
        metric.use_sample_weights = False
        res = metric.process()
        for k, v in res.items():
            print(k, '\t', v)
