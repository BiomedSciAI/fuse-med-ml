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

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Hashable
import traceback

import numpy as np

from fuse.eval.metrics.metrics_common import MetricDefault, MetricWithCollectorBase
from fuse.eval.metrics.libs.classification import MetricsLibClass


class MetricMultiClassDefault(MetricWithCollectorBase):
    """
    Default generic implementation for metric
    Can be used for any metric getting as an input list of prediction, list of targets and optionally additional parameters
    """

    def __init__(
        self,
        pred: str,
        target: str,
        metric_func: Callable,
        class_names: Optional[Sequence[str]] = None,
        class_weights: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        """
        :param pred: prediction key to collect
        :param target: target key to collect
        :param metric_func: function getting as a input list of predictions, targets and optionally more arguments specified in kwargs
                            the function should return a result or a dictionary of results
        :param class_names: class names for multi-class evaluation or None for binary evaluation
        :param class_weight: weight per class - the macro_average result will be a weighted sum rather than an average
        :param kwargs: additional kw arguments for MetricWithCollectorBase
        """
        super().__init__(pred=pred, target=target, **kwargs)
        self._metric_func = metric_func
        self._class_names = class_names
        self._class_weights = class_weights

    def eval(
        self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None
    ) -> Union[Dict[str, Any], Any]:
        """
        See super class
        """
        # extract values from collected data and results dict
        kwargs = self._extract_arguments(results, ids)

        if self._class_names is None:
            # single evaluation for all classes at once / binary classifier
            try:
                metric_results = self._metric_func(**kwargs)
            except:
                track = traceback.format_exc()
                print(f"Error in metric: {track}")
                metric_results = None
        else:
            metric_results = {}

            # one vs rest evaluation per class, including average
            try:
                # compute one-vs-rest metrics
                is_dict = False
                all_classes = []
                for cls_index, cls_name in enumerate(self._class_names):
                    cls_res = self._metric_func(pos_class_index=cls_index, **kwargs)
                    if isinstance(cls_res, dict):
                        is_dict = True
                        for sub_metric_name in cls_res:
                            metric_results[f"{sub_metric_name}.{cls_name}"] = cls_res[sub_metric_name]
                    else:
                        assert (
                            is_dict is False
                        ), "expect all sub metric results to either return dictionary or single value"
                        metric_results[f"{cls_name}"] = cls_res

                    all_classes.append(cls_res)

                # compute macro average
                if is_dict:
                    for key in all_classes[0]:
                        all_classes_elem = np.array([d[key] for d in all_classes])
                        indices = ~np.isnan(all_classes_elem)
                        if self._class_weights is None:
                            weights = None
                        else:
                            weights = self._class_weights[indices]
                        metric_results[f"{key}.macro_avg"] = np.average(all_classes_elem[indices], weights=weights)
                else:
                    all_classes = np.array(all_classes)
                    indices = ~np.isnan(all_classes)
                    if self._class_weights is None:
                        weights = None
                    else:
                        weights = self._sum_weights[indices]
                    metric_results["macro_avg"] = np.average(all_classes[indices], weights=weights)
            except:
                track = traceback.format_exc()
                print(f"Error in metric: {type(self).__name__} - {track}")
                for cls_name in self._class_names:
                    metric_results[f"{cls_name}"] = None
                metric_results["macro_avg"] = None

        return metric_results


class MetricAUCROC(MetricMultiClassDefault):
    """
    Compute auc roc (Receiver operating characteristic) score using sklearn (one vs rest)
    """

    def __init__(
        self,
        pred: str,
        target: str,
        class_names: Optional[Sequence[str]] = None,
        max_fpr: Optional[float] = None,
        **kwargs,
    ):
        """
        See MetricMultiClassDefault for the missing params
        :param max_fpr: float > 0 and <= 1, default=None
                        If not ``None``, the standardized partial AUC over the range [0, max_fpr] is returned.
        """
        auc_roc = partial(MetricsLibClass.auc_roc, max_fpr=max_fpr)
        super().__init__(pred, target, metric_func=auc_roc, class_names=class_names, **kwargs)


class MetricROCCurve(MetricDefault):
    """
    Output the roc curve (Multi class version - one vs rest)
    """

    def __init__(
        self,
        pred: str,
        target: str,
        output_filename: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        sample_weight: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        :param pred:                key for predicted output (e.g., class scores after softmax)
        :param target:              key for target (e.g., ground truth label)
        :param output_filename:     output filename to save the figure. if None, will just save the arguments to create the figure in results dictionary.
        :param class_names:         Name for each class otherwise will assume binary classification and create curve just for class 1.
        :param sample_weight:       Optional, key for  sample weight
        :param kwargs:              super class arguments
        """
        roc = partial(MetricsLibClass.roc_curve, class_names=class_names, output_filename=output_filename)
        super().__init__(pred=pred, target=target, sample_weight=sample_weight, metric_func=roc, **kwargs)


class MetricAUCPR(MetricMultiClassDefault):
    """
    Compute Area Under Precision Recall Curve score using sklearn (one vs rest)
    """

    def __init__(self, pred: str, target: str, class_names: Optional[Sequence[str]] = None, **kwargs):
        super().__init__(pred, target, MetricsLibClass.auc_pr, class_names=class_names, **kwargs)


class MetricAccuracy(MetricDefault):
    """
    Compute accuracy over all the samples
    """

    def __init__(self, pred: str, target: str, sample_weight: Optional[str] = None, **kwargs):
        """
        See MetricDefault for the missing params
        :param sample_weight: weight per sample for the final accuracy score. Keep None if not required.
        """
        super().__init__(
            pred=pred, target=target, sample_weight=sample_weight, metric_func=MetricsLibClass.accuracy, **kwargs
        )


class MetricConfusion(MetricMultiClassDefault):
    """
    Compute metrics derived from one-vs-rest confusion matrix such as 'sensitivity', 'recall', 'tpr', 'specificity',  'selectivity', 'npr', 'precision', 'ppv', 'f1'
    """

    def __init__(
        self,
        pred: str,
        target: str,
        class_names: Optional[Sequence[str]] = None,
        metrics: Sequence[str] = ("sensitivity",),
        operation_point: Union[float, Sequence[Tuple[int, float]], str, None] = tuple(),
        **kwargs,
    ):
        """
        See MetricMultiClassDefault for the missing params
        :param metrics: sequence of required metrics that dervied from confusion matrix.
                        Options: 'sensitivity', 'recall', 'tpr', 'specificity',  'selectivity', 'npr', 'precision', 'ppv', 'f1'
        """
        super().__init__(
            pred=pred,
            target=target,
            metric_func=MetricsLibClass.confusion_metrics,
            class_names=class_names,
            metrics=metrics,
            **kwargs,
        )
        self._metrics = metrics


class MetricConfusionMatrix(MetricDefault):
    """
    Computes the confusion matrix from the data
    Multi class version
    returns dataframe
    """

    def __init__(
        self, cls_pred: str, target: str, class_names: Sequence[str], sample_weight: Optional[str] = None, **kwargs
    ) -> None:
        """
        See super class
        :param cls_pred: key to class predictions
        """
        conf_matrix_func = partial(MetricsLibClass.confusion_matrix, class_names=class_names)
        super().__init__(
            pred=None,
            cls_pred=cls_pred,
            target=target,
            sample_weight=sample_weight,
            metric_func=conf_matrix_func,
            **kwargs,
        )


class MetricBSS(MetricDefault):
    """
    Multi class version for Brier Skill Score (BSS)
    bss = 1 - bs / bs_{ref}

    bs_{ref} will be computed for a model that makes a predictions according to the prevalance of each class in dataset

    """

    def __init__(self, pred: str, target: str, **kwargs):
        """
        See super class
        """
        super().__init__(pred=pred, target=target, metric_func=MetricsLibClass.multi_class_bss, **kwargs)
