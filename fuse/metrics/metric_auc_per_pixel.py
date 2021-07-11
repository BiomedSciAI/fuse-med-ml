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
from typing import Dict, Optional, Callable

import numpy as np
import sklearn.metrics
import torch
from fuse.metrics.metric_base import FuseMetricBase


def make_one_hot(input, num_classes, device='cuda'):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    if len(input.shape) < 4:
        input = input.unsqueeze(0).unsqueeze(0)
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.long(), 1)
    result = result.squeeze(0).cpu().numpy()
    return result


class FuseMetricAUCPerPixel(FuseMetricBase):
    """
    Area under receiver operating characteristic curve
    Multi class version
    Will add to epoch results:
    metrics.auc.<cls> - auc per class vs rest - where <cls> will be the class name if supplied or class index
    metric.auc.avg - average of all class vs rest results
    """

    def __init__(self, pred_name: str, target_name: str, filter_func: Optional[Callable] = None, class_names: Optional[str]=None):
        """
        :param pred_name:       batch_dict key for predicted output (e.g., class probabilities after softmax)
        :param target_name:     batch_dict key for target (e.g., ground truth label)
        :param filter_func: function that filters batch_dict/ The function gets ans input batch_dict and returns filtered batch_dict
        :param class_names:     Optional - name for each class otherwise the class index will be used.
        """
        super().__init__(pred_name, target_name, filter_func)
        self._class_names = class_names

    def process(self) -> Dict[str, float]:
        """
        Calculates ROCAUC per epoch
        :return: dictionary including Area under ROC curve (floating point in range [0, 1]/-1) for each class (class vs rest),
                 -1 will be set for invalid/undefined result (an error message will be printed)
                 The dictionary will also include the average AUC
        """
        results = {}
        aucs_per_class = []
        if self._class_names is None:
            num_classes = self.epoch_preds[0].shape[0]
            class_names = ['class_' + str(i) for i in range(num_classes)]
        else:
            class_names = self._class_names
        try:
            n_classes = len(class_names)

            for cls_ind, cls in enumerate(class_names):
                predictions = []
                targets = []
                for pred, targ in zip(self.epoch_preds, self.epoch_targets):
                    pred_siz = pred.shape

                    # Convert target to one hot encoding
                    if n_classes > 1 and targ.ndim == 2:
                        targ = make_one_hot(torch.from_numpy(targ), n_classes)
                    predictions.extend(pred[cls_ind].reshape(pred_siz[1]*pred_siz[2]))
                    targets.extend(targ[cls_ind].reshape(pred_siz[1]*pred_siz[2]))

                fpr, tpr, _ = sklearn.metrics.roc_curve(targets, predictions, pos_label=1)
                auc = sklearn.metrics.auc(fpr, tpr)
                results[cls] = auc
                aucs_per_class.append(auc)

            results['macro_avg'] = np.nanmean(aucs_per_class)
        except Exception as e :
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
            'targets': np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])}

    data = {'preds': np.array([[[0.8, 0.1, 0.1],
                               [0.2, 0.5, 0.1]],
                               [[0.8, 0.1, 0.1],
                                [0.2, 0.5, 0.1]],
                               [[0.8, 0.1, 0.1],
                                [0.2, 0.5, 0.1]],
                               [[0.8, 0.1, 0.1],
                                [0.2, 0.5, 0.1]],
                               [[0.8, 0.1, 0.1],
                                [0.2, 0.5, 0.1]],
                               [[0.8, 0.1, 0.1],
                                [0.2, 0.5, 0.1]],
                               [[0.8, 0.1, 0.1],
                                [0.2, 0.5, 0.1]],
                               [[0.8, 0.1, 0.1],
                                [0.2, 0.5, 0.1]],
                               [[0.8, 0.1, 0.1],
                                [0.2, 0.5, 0.1]],
                               [[0.8, 0.1, 0.1],
                                [0.2, 0.5, 0.1]]]),
            'targets': np.array([[[1, 0, 0],
                                  [0, 0, 0]],
                                 [[1, 0, 0],
                                  [0, 0, 0]],
                                 [[1, 0, 0],
                                  [0, 0, 0]],
                                 [[1, 0, 0],
                                  [0, 0, 0]],
                                 [[1, 0, 0],
                                  [0, 0, 0]],
                                 [[1, 0, 0],
                                  [0, 0, 0]],
                                 [[1, 0, 0],
                                  [0, 0, 0]],
                                 [[1, 0, 0],
                                  [0, 0, 0]],
                                 [[1, 0, 0],
                                  [0, 0, 0]],
                                 [[1, 0, 0],
                                  [0, 0, 0]]])}

    metric = FuseMetricAUCPerPixel(pred_name='preds', target_name='targets')
    metric.collect(data)
    res = metric.process()
    for k, v in res.items():
        print(k, '\t', v)
