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

import numpy as np
from sklearn import metrics

from fuse.metrics.classification.metric_default import FuseMetricDefault


class FuseMetricAUC(FuseMetricDefault):
    """
    Area under the receiver operating characteristic curve
    Multi class version
    """

    def __init__(self,
                 **kwargs) -> None:
        """
        See FuseMetricDefault
        """
        super().__init__(metrics.roc_auc_score, **kwargs)


class FuseMetricAUCPR(FuseMetricDefault):
    """
    Area under the precision recall curve
    Multi class version
    """

    def __init__(self,
                 **kwargs) -> None:
        """
        See FuseMetricDefault
        """
        super().__init__(aucpr, **kwargs)


def aucpr(y_true:np.ndarray, y_score:np.ndarray, sample_weight:np.ndarray):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score,
                                                          sample_weight=sample_weight)
    return metrics.auc(recall, precision)


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
            'weights': np.array([0.03, 0.12, 0.05, 0.52, 0.23, 0.72, 0.13, 0.113, 0.84, 0.09])}

    metric_dict = {"auc": FuseMetricAUC(pred_name='preds', target_name='targets', use_sample_weights=True,
                                        sample_weight_name='weights'),
                   "aucpr": FuseMetricAUCPR(pred_name='preds', target_name='targets', use_sample_weights=True,
                                            sample_weight_name='weights')}
    for metric_name, metric in metric_dict.items():
        metric.collect(data)
        res = metric.process()
        print("use_sample_weights = True")
        for k, v in res.items():
            print(k, '\t', v)

        print("use_sample_weights = False")
        metric.use_sample_weights = False

        res = metric.process()

        for k, v in res.items():
            print(k, '\t', v)
