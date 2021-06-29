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

from Fuse.metrics.metric_base import FuseMetricBase


class FuseMetricPartialAUC(FuseMetricBase):
    """
    Partial area under receiver operating characteristic curve
    Multi class version
    Will add to epoch results:
    metrics.<metric_name>.<cls> - partial auc per class vs rest - where <cls> will be the class name if supplied or class index
    metric.<metric_name>.macro_avg - average of all class vs rest results
    """

    def __init__(self,
                 pred_name: str,
                 target_name: str,
                 tpr_val: float,
                 class_names: Optional[List[str]] = None,
                 filter_func: Optional[Callable] = None,
                 use_sample_weights: Optional[bool] = False,
                 sample_weight_name: Optional[str] = None) -> None:
        """
        :param pred_name:           batch_dict key for predicted output (e.g., class probabilities after softmax)
        :param target_name:         batch_dict key for target (e.g., ground truth label)
        :param tpr_val:             the minimum value of sensitivity to compute partial AUC
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
        self._tpr_val = tpr_val

    def process(self) -> Dict[str, float]:
        """
        Calculates partial ROCAUC per epoch
        :return: dictionary including Partial Area under ROC curve (floating point in range [0, 1]/-1) for each class (class vs rest),
                 -1 will be set for invalid/undefined result (an error message will be printed)
                 The dictionary will also include the average Partial AUC
        """
        results = {}
        paucs_per_class = []
        if self._class_names is None:
            num_classes = self.collected_data['pred'][0].shape[0]
            class_names = ['class_' + str(i) for i in range(num_classes)]
        else:
            class_names = self._class_names
        sample_weight = self.collected_data['sample_weight'] if self.use_sample_weights else None
        try:
            for cls_ind, cls in enumerate(class_names):
                predicted_ind = [pred[cls_ind] for pred in self.collected_data['pred']]
                fpr, tpr, _ = sklearn.metrics.roc_curve(self.collected_data['target'], predicted_ind, pos_label=cls_ind, sample_weight=sample_weight)
                truncated_tprs = np.maximum(0, tpr - self._tpr_val)
                pauc = sklearn.metrics.auc(fpr, truncated_tprs)
                pauc = pauc / (1 - self._tpr_val)  # normalize
                results[cls] = pauc
                paucs_per_class.append(pauc)

            results['macro_avg'] = np.nanmean(paucs_per_class)
        except:
            track = traceback.format_exc()
            logging.getLogger('Fuse').error(track)
            results = {cls: -1 for cls in class_names}
            results['macro_avg'] = -1
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

    metric = FuseMetricPartialAUC(pred_name='preds', target_name='targets', tpr_val=0.95, use_sample_weights=True,
                           sample_weight_name='weights')
    metric.collect(data)
    res = metric.process()
    print("use_sample_weights = True")
    for k, v in res.items():
        print(k, '\t', v)

    metric = FuseMetricPartialAUC(pred_name='preds', target_name='targets', tpr_val=0.95)
    metric.collect(data)
    res = metric.process()
    print("use_sample_weights = False")
    for k, v in res.items():
        print(k, '\t', v)