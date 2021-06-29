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
from typing import Sequence, Tuple, List, Union, Optional, Any

import numpy as np
import sklearn
from sklearn.utils import compute_sample_weight
import  matplotlib.pyplot as plt

class FuseMetricsToolBox:
    """
    Common utils for metrics
    """
    @staticmethod
    def convert_probabilities_to_class(prediction: Union[List[np.ndarray], np.ndarray], thresholds: List[Tuple] = None) -> np.array:
        """
        convert probabilities to class prediction, threshold per class
        :param prediction: either list of numpy arrays of shape [NUM_CLASSES] or numpy array of shape [N, NUM_CLASSES]
        :param thresholds: list of tuples (class_idx, threshold)
        :return: array of classes
        """
        # if no threshold specified, simply apply argmax
        if thresholds is None:
            return np.argmax(np.array(prediction), -1)

        # convert according to thresholds
        output_class = np.array([-1 for x in range(len(prediction))])
        for thr_idx, thr in enumerate(thresholds):
            class_idx = thr[0]
            class_thr = thr[1]

            # if it is the last one, set the rest of the samples with this class
            if thr_idx == (len(thresholds) - 1):
                return np.where(output_class == -1, class_idx, output_class)

            # among all the samples which not already predicted, set the ones that cross the threshold with this class
            target_idx = np.argwhere(np.logical_and(np.array(prediction)[:, class_idx] > class_thr, np.array(output_class) == -1))
            output_class[target_idx] = class_idx

        return output_class

    @staticmethod
    def find_operation_points(predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray], objectives:Sequence[Tuple[int, str, float]]) -> Sequence[Tuple[float]]:
        """
        Multi class version to find operation point in classification task
        :param predictions: List of arrays of shape [NUM_CLASSES]
        :param targets: List of arrays specifying the target class per
        :param objectives: Sequence of tuples, each element including (class_index, 'sensitivity' / 'specificity', value)
        :return:
        """
        operating_point_thresholds: Sequence[Tuple[int, float]] = []
        already_predicted_indices: List[int] = []

        for op in objectives:
            cls = op[0]

            # get the predicted probability for a specific class.
            # Consider all the samples that already predicted as previous classes as probability 0
            predicted_cls = []
            for i in range(len(predictions)):
                if i in already_predicted_indices:
                    predicted_cls.append(0.0)
                else:
                    predicted_cls.append(predictions[i][cls])

            # get roc curve
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(np.array(targets), np.array(predicted_cls), pos_label=cls)

            # find operating point given the objective and target
            objective_str = op[1]
            assert objective_str in ["sensitivity", "specificity", None]
            if objective_str is not None:
                target = op[2]
                if objective_str == "sensitivity":
                    objective = tpr - target
                elif objective_str == "specificity":
                    objective = 1 - fpr - target
                objective[objective < 0.0] = 1.0
                operating_point_index = np.max(np.where(objective.min() == objective))
                threshold = thresholds[operating_point_index]

                # mark all the samples that cross this as threshold as already predicted.
                predicted = [i for i, val in enumerate(thresholds) if val > threshold]
                already_predicted_indices.extend(predicted)

                # add this operating point to the returned list
                operating_point_thresholds.append((cls, threshold))
            else:
                # in  a case of None objective all the rest of the samples prediction will be that specific class
                operating_point_thresholds.append((cls, 0.0))

        return operating_point_thresholds

    @staticmethod
    def confusion_metrics(prediction: np.array, cls_ind:int, target:np.array, metrics:Sequence[str] = tuple(), class_weight: Any = None)->Tuple[float]:
        """
        Compute confusion metrics
        :param prediction: class predictions
        :param cls_ind: the class to compute the metrics for in one vs all manner
        :param target: the target classes
        :param metrics: required metrics names, options: 'sensitivity', 'recall', 'tpr', 'specificity',  'selectivity', 'npr', 'precision', 'ppv', 'f1'
        :param class_weight: use None to avoid form class weight or see available options in sklearn.utils.compute_sample_weight()
        :return: dictionary, including the computed values for the required metrics
        """
        if class_weight is not None:
            sample_weight = compute_sample_weight(class_weight=class_weight, y=target)
        else:
            sample_weight = np.ones_like(target)

        class_target_t = np.where(target == cls_ind, 1, 0)
        class_pred_t = np.where(prediction == cls_ind, 1, 0)
        res = {}
        res['tp'] = (np.logical_and(class_target_t, class_pred_t)*sample_weight).sum()
        res['fn'] = (np.logical_and(class_target_t, np.logical_not(class_pred_t))*sample_weight).sum()
        res['fp'] = (np.logical_and(np.logical_not(class_target_t), class_pred_t)*sample_weight).sum()
        res['tn'] = (np.logical_and(np.logical_not(class_target_t), np.logical_not(class_pred_t))*sample_weight).sum()

        for metric in metrics:
            if metric in ['sensitivity', 'recall', 'tpr']:
                res[metric] = res['tp'] / (res['tp'] + res['fn'])
            elif metric in ['specificity', 'selectivity', 'tnr']:
                res[metric] = res['tn'] / (res['tn'] + res['fp'])
            elif metric in ['precision', 'ppv']:
                res[metric] = res['tp'] / (res['tp'] + res['fp'])
            elif metric in ['f1']:
                res[metric] = 2 * res['tp'] / (2 * res['tp'] + res['fp'] + res['fn'])
            else:
                msg = f'unknown metric {metric}'
                logging.getLogger('Fuse').error(msg)
                raise Exception(msg)

        return res

    @staticmethod
    def get_class_names(num_classes: int, class_names: Optional[Sequence[str]] = None) -> List[str]:
        """
        Get class_names either automatically generated by the given the number of classes or keep the ones provided by the user
        :param num_classes: number of classes
        :param class_names: Optional, option for the user to specify the class names
        :return: list of class names
        """
        if class_names is None:
            num_classes = num_classes
            res_class_names = ['class_' + str(i) for i in range(num_classes)]
        else:
            res_class_names = class_names

        return res_class_names

    @staticmethod
    def roc_curve(predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray], samples_weight: Sequence[np.ndarray],  class_names: str, output_filename: str) -> None:
        """
        Multi class version for roc curve
        :param predictions: List of arrays of shape [NUM_CLASSES]
        :param targets: List of arrays specifying the target class per
        :return:
        """
        for cls, cls_name in enumerate(class_names):
            fpr, tpr, _ = sklearn.metrics.roc_curve(targets, np.array(predictions)[:, cls], sample_weight=samples_weight, pos_label=cls)
            auc = sklearn.metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{cls_name}({auc:0.2f})')

        plt.title("ROC curve")
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.savefig(output_filename)
        plt.close()