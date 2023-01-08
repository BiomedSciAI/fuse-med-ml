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

from fuse.eval.metrics.classification.metrics_classification_common import MetricMultiClassDefault

from ehrtransformers.model import metrics

class FuseMetricAUC(MetricMultiClassDefault):
    """
    Basic AUC metric - Operation point is not used -
    the predicted class will simply be the one with the max probability
    """

    def __init__(self,
                 pred: str,
                 target: str,
                 **kwargs):
        """
        See MetricMultiClassDefault for the missing params
        :param max_fpr: float > 0 and <= 1, default=None
                        If not ``None``, the standardized partial AUC over the range [0, max_fpr] is returned.
        """
        # metrics.auroc first removes the rows in which target is the same for all labels, and then calculates AUC
        super().__init__(pred=None, target=None, logits=pred, label=target, metric_func=metrics.auroc, **kwargs)


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
            'targets': np.array([[0, 1, 0],
                               [0, 0, 1],
                               [1, 1, 0],
                               [0, 1, 0],
                               [0, 0, 1],
                               [0, 1, 1],
                               [0, 0, 0],
                               [1, 1, 1],
                               [0, 1, 0],
                               [0, 0, 1]])}

    metric = FuseMetricAUC(pred='preds', target='targets')
    metric.collect(data)
    res = metric.eval()
    print(res)

