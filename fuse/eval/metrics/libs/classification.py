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
from typing import Dict, Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np

from sklearn import metrics
import sklearn
from sklearn.utils.multiclass import type_of_target


import matplotlib.pyplot as plt


class MetricsLibClass:
    @staticmethod
    def auc_roc(
        pred: Sequence[Union[np.ndarray, float]],
        target: Sequence[Union[np.ndarray, int]],
        sample_weight: Optional[Sequence[Union[np.ndarray, float]]] = None,
        pos_class_index: int = -1,
        max_fpr: Optional[float] = None,
    ) -> float:
        """
        Compute auc roc (Receiver operating characteristic) score using sklearn (one vs rest)
        :param pred: prediction array per sample. Each element shape [num_classes]
        :param target: target per sample. Each element is an integer in range [0 - num_classes)
        :param sample_weight: Optional - weight per sample for a weighted auc. Each element is  float in range [0-1]
        :param pos_class_index: the class to compute the metrics in one vs rest manner - set to 1 in binary classification
        :param max_fpr: float > 0 and <= 1, default=None
                        If not ``None``, the standardized partial AUC over the range [0, max_fpr] is returned.
        :return auc Receiver operating characteristic score
        """
        if not isinstance(pred[0], np.ndarray):
            pred = [np.array(p) for p in pred]
            pos_class_index = 1
            y_score = np.asarray(pred)
        else:
            if pos_class_index < 0:
                pos_class_index = pred[0].shape[0] - 1
            y_score = np.asarray(pred)[:, pos_class_index]

        return metrics.roc_auc_score(
            y_score=y_score, y_true=np.asarray(target) == pos_class_index, sample_weight=sample_weight, max_fpr=max_fpr
        )

    @staticmethod
    def roc_curve(
        pred: Sequence[Union[np.ndarray, float]],
        target: Sequence[Union[np.ndarray, int]],
        class_names: Sequence[str],
        sample_weight: Optional[Sequence[Union[np.ndarray, float]]] = None,
        output_filename: Optional[str] = None,
    ) -> Dict:
        """
        Multi class version for roc curve
        :param pred: List of arrays of shape [NUM_CLASSES]
        :param target: List of arrays specifying the target class per sample
        :return: saving roc curve to a file and return the input to figure to a dictionary
        """
        # if class_names not specified assume binary classification
        if class_names is None:
            class_names = [None, "Positive"]
        # extract info for the plot
        results = {}
        for cls, cls_name in enumerate(class_names):
            if cls_name is None:
                continue
            fpr, tpr, _ = sklearn.metrics.roc_curve(
                target, np.array(pred)[:, cls], sample_weight=sample_weight, pos_label=cls
            )
            auc = sklearn.metrics.auc(fpr, tpr)
            results[cls_name] = {"fpr": fpr, "tpr": tpr, "auc": auc}

        # display
        if output_filename is not None:
            for cls_name, cls_res in results.items():
                plt.plot(cls_res["fpr"], cls_res["tpr"], label=f'{cls_name}(auc={cls_res["auc"]:0.2f})')

            plt.title("ROC curve")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            plt.savefig(output_filename)
            plt.close()

        return results

    @staticmethod
    def auc_pr(
        pred: Sequence[Union[np.ndarray, float]],
        target: Sequence[Union[np.ndarray, int]],
        sample_weight: Optional[Sequence[Union[np.ndarray, float]]] = None,
        pos_class_index: int = -1,
    ) -> float:
        """
        Compute auc pr (precision-recall) score using sklearn (one vs rest)
        :param pred: prediction array per sample. Each element shape [num_classes]
        :param target: target per sample. Each element is an integer in range [0 - num_classes)
        :param sample_weight: Optional - weight per sample for a weighted auc. Each element is  float in range [0-1]
        :param pos_class_index: the class to compute the metrics in one vs rest manner - set to 1 in binary classification
        :return auc precision recall score
        """
        if not isinstance(pred[0], np.ndarray):
            pred = [np.array(p) for p in pred]
            pos_class_index = 1
            y_score = np.asarray(pred)
        else:
            if pos_class_index < 0:
                pos_class_index = pred[0].shape[0] - 1
            y_score = np.asarray(pred)[:, pos_class_index]

        precision, recall, _ = metrics.precision_recall_curve(
            probas_pred=y_score, y_true=np.asarray(target) == pos_class_index, sample_weight=sample_weight
        )
        return metrics.auc(recall, precision)

    @staticmethod
    def accuracy(
        pred: Sequence[Union[np.ndarray, int]],
        target: Sequence[Union[np.ndarray, int]],
        sample_weight: Optional[Sequence[Union[np.ndarray, float]]] = None,
    ):
        """
        Compute accuracy score
        :param pred: class prediction. Each element is an integer in range [0 - num_classes) or a float in range [0-1] in which case argmax will be applied
        :param target: the target class. Each element is an integer in range [0 - num_classes) or its one hot encoded version
        :param sample_weight: Optional - weight per sample for a weighted score. Each element is  float in range [0-1]
        :return: accuracy score
        """

        pred = np.array(pred)
        target = np.array(target)

        if type_of_target(pred) in ("continuous", "continuous-multioutput"):
            pred = np.argmax(pred, -1)
        if type_of_target(target) in ("multilabel-indicator", "multiclass-multioutput"):
            target = np.argmax(target, -1)

        return metrics.accuracy_score(target, pred, sample_weight=sample_weight)

    @staticmethod
    def confusion_metrics(
        pred: Sequence[Union[np.ndarray, int]],
        target: Sequence[Union[np.ndarray, int]],
        pos_class_index: int = 1,
        metrics: Sequence[str] = tuple(),
        sample_weight: Optional[Sequence[Union[np.ndarray, float]]] = None,
    ) -> Dict[str, float]:
        """
        Compute metrics derived from one-vs-rest confusion matrix such as 'sensitivity', 'recall', 'tpr', 'specificity',  'selectivity', 'npr', 'precision', 'ppv', 'f1'
        Assuming that there are positive cases and negative cases in targets
        :param pred: class prediction. Each element is an integer in range [0 - num_classes) or a float in range [0-1] in which case argmax will be applied
        :param target: the target class. Each element is an integer in range [0 - num_classes) or its one hot encoded version
        :param pos_class_index: the class to compute the metrics in one vs rest manner - set to 1 in binary classification
        :param metrics: required metrics names, options: 'sensitivity', 'recall', 'tpr', 'specificity',  'selectivity', 'npr', 'precision', 'ppv', 'f1'
        :param sample_weight: Optional - weight per sample for a weighted score. Each element is  float in range [0-1]
        :return: dictionary, including the computed values for the required metrics.
                 format: {"tp": <>, "tn": <>, "fp": <>, "fn": <>, <required metric name>: <>}
        """
        pred = np.array(pred)
        target = np.array(target)

        if type_of_target(pred) in ("continuous", "continuous-multioutput"):
            pred = np.argmax(pred, -1)
        if type_of_target(target) in ("multilabel-indicator", "multiclass-multioutput"):
            target = np.argmax(target, -1)

        class_target_t = np.where(target == pos_class_index, 1, 0)
        class_pred_t = np.where(pred == pos_class_index, 1, 0)
        if sample_weight is None:
            sample_weight = np.ones_like(class_target_t)

        res = {}
        tp = (np.logical_and(class_target_t, class_pred_t) * sample_weight).sum()
        fn = (np.logical_and(class_target_t, np.logical_not(class_pred_t)) * sample_weight).sum()
        fp = (np.logical_and(np.logical_not(class_target_t), class_pred_t) * sample_weight).sum()
        tn = (np.logical_and(np.logical_not(class_target_t), np.logical_not(class_pred_t)) * sample_weight).sum()

        for metric in metrics:
            if metric in ["sensitivity", "recall", "tpr"]:
                res[metric] = tp / (tp + fn)
            elif metric in ["specificity", "selectivity", "tnr"]:
                res[metric] = tp / (tn + fp)
            elif metric in ["precision", "ppv"]:
                if tp + fp != 0:
                    res[metric] = tp / (tp + fp)
                else:
                    res[metric] = 0
            elif metric in ["f1"]:
                res[metric] = 2 * tp / (2 * tp + fp + fn)
            elif metric in ["matrix"]:
                res["tp"] = tp
                res["fn"] = fn
                res["fp"] = fp
                res["tn"] = tn
            else:
                raise Exception(f"unknown metric {metric}")

        return res

    @staticmethod
    def confusion_matrix(
        cls_pred: Sequence[int],
        target: Sequence[int],
        class_names: Sequence[str],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates Confusion Matrix (multi class version)
        :param cls_pred: sequence of class prediction or a floats in range [0-1]
        :param target: sequence of labels or their one hot encoded versions
        :param class_names: string name per class
        :param sample_weight: optional, weight per sample.
        :return: {"count": <confusion matrix>, "percent" : <confusion matrix - percent>)
        Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
        """

        if type_of_target(cls_pred) in ("continuous", "continuous-multioutput"):
            cls_pred = np.argmax(cls_pred, -1)
        if type_of_target(target) in ("multilabel-indicator", "multiclass-multioutput"):
            target = np.argmax(target, -1)

        conf_matrix = sklearn.metrics.confusion_matrix(y_true=target, y_pred=cls_pred, sample_weight=sample_weight)
        conf_matrix_count = pd.DataFrame(conf_matrix, columns=class_names, index=class_names)
        conf_matrix_total = conf_matrix.sum(axis=1)
        conf_matrix_count["total"] = conf_matrix_total
        conf_matrix_percent = pd.DataFrame(
            conf_matrix / conf_matrix_total[:, None], columns=class_names, index=class_names
        )

        return {"count": conf_matrix_count, "percent": conf_matrix_percent}

    @staticmethod
    def multi_class_bs(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Brier Score:
        bs = 1/N * SUM_n SUM_c (pred_{n,c} - target_{n,c})^2
        :param pred: probability score. Expected Shape [N, C]
        :param target: target class (int) per sample. Expected Shape [N]
        """
        # create one hot vector
        target_one_hot = np.zeros_like(pred)
        target_one_hot[np.arange(target_one_hot.shape[0]), target] = 1

        return float(np.mean(np.sum((pred - target_one_hot) ** 2, axis=1)))

    @staticmethod
    def multi_class_bss(pred: Sequence[np.ndarray], target: Sequence[np.ndarray]) -> float:
        """
        Brier Skill Score:
        bss = 1 - bs / bs_{ref}

        bs_{ref} will be computed for a model that makes a predictions according to the prevalance of each class in dataset

        :param pred: probability score. Expected Shape [N, C]
        :param target: target class (int) per sample. Expected Shape [N]
        """
        if isinstance(pred[0], np.ndarray) and pred[0].shape[0] > 1:
            pred = np.array(pred)
        else:
            # binary case
            pred = np.array(pred)
            pred = np.stack((1 - pred, pred), axis=-1)
        target = np.array(target)

        # BS
        bs = MetricsLibClass.multi_class_bs(pred, target)

        # no skill BS
        no_skill_prediction = [(target == target_cls).sum() / target.shape[0] for target_cls in range(pred.shape[-1])]
        no_skill_predictions = np.tile(np.array(no_skill_prediction), (pred.shape[0], 1))
        bs_ref = MetricsLibClass.multi_class_bs(no_skill_predictions, target)

        return 1.0 - bs / bs_ref

    @staticmethod
    def convert_probabilities_to_class(
        pred: Sequence[Union[np.ndarray, float]], operation_point: Union[float, Sequence[Tuple[int, float]]]
    ) -> np.array:
        """
        convert probabilities to class prediction
        :param pred: sequence of numpy arrays / floats of shape [NUM_CLASSES]
        :param operation_point: list of tuples (class_idx, threshold) or empty sequence for argmax
        :return: array of class predictions
        """
        if isinstance(pred[0], np.ndarray) and pred[0].shape[0] > 1:
            pred = np.array(pred)
        else:
            # binary case
            pred = np.array(pred)
            pred = np.stack((1 - pred, pred), axis=-1)

        # if no threshold specified, simply apply argmax
        if operation_point is None or (isinstance(operation_point, Sequence) and len(operation_point) == 0):
            return np.argmax(pred, -1)

        # binary operation point
        if isinstance(operation_point, float):
            if pred[0].shape[0] == 2:
                return np.where(pred[:, 1] > operation_point, 1, 0)
            elif pred[0].shape[0] == 1:
                return np.where(pred > operation_point, 1, 0)
            else:
                raise Exception("Error - got single float as an operation point for multi-class prediction")

        # convert according to thresholds
        output_class = np.array([-1 for x in range(len(pred))])
        for thr in operation_point:
            class_idx = thr[0]
            class_thr = thr[1]
            # argmax
            if class_idx == "argmax":
                output_class[output_class == -1] = np.argmax(pred, -1)[output_class == -1]

            # among all the samples which not already predicted, set the ones that cross the threshold with this class
            target_idx = np.argwhere(np.logical_and(pred[:, class_idx] > class_thr, output_class == -1))
            output_class[target_idx] = class_idx

        return output_class
