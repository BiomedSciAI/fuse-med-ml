from functools import partial
from typing import Dict, List, Optional
from collections import defaultdict
from fuse.eval.metrics.libs.segmentation import MetricsSegmentation

import numpy as np

from fuse.eval.metrics.metrics_common import MetricPerSampleDefault


def average_sample_results(
    metric_result: List[Dict[int, float]], class_weights: Optional[Dict[int, float]] = None
) -> Dict[str, float]:
    """
    Calculates average result per class and average result over classes on a specific metric
    metric_result assumed to have same type of keys as in class_weights which represents the different classes
    :param metric_result: list of per image metric results ,each element is a dictionary where the key is class id and value is the metric score
    :param class_weights: weight per segmentation class , we assume sum of total weights is 1 and each element is in 0-1 range
    :return: dictionary of average result per class and average result over classes
    """
    average_results = {}
    aggregated_results = defaultdict(list)
    for sample in metric_result:
        for key, score in sample.items():
            aggregated_results[key].append(score)
    total_avarage = 0
    if class_weights is None:
        for key, result_list in aggregated_results.items():
            average_results[key] = np.sum(result_list) / len(result_list)
            total_avarage += average_results[key]
    else:
        for key, result_list in aggregated_results.items():
            average_results[key] = np.sum(result_list) / len(result_list)
            total_avarage += class_weights[key] * average_results[key]
    average_results["average"] = total_avarage / len(aggregated_results)
    return average_results


class MetricDice(MetricPerSampleDefault):
    """
    Compute similarity dice score (2*|X&Y| / (|X|+|Y|)) for every label
    """

    def __init__(
        self,
        pred: str,
        target: str,
        pixel_weight: Optional[str] = None,
        class_weights: Optional[Dict[int, float]] = None,
        **kwargs
    ):
        """
        See super class for the missing params
        used for sematric and binary segmentation
        :param class_weights: weight per segmentation class , we assume sum of total weights is 1 and each element is in 0-1 range
        :param pixel_weight: Optional dictionary key to collect
        """
        average = partial(average_sample_results, class_weights=class_weights)
        super().__init__(
            pred=pred,
            target=target,
            pixel_weight=pixel_weight,
            metric_per_sample_func=MetricsSegmentation.dice,
            result_aggregate_func=average,
            **kwargs
        )


class MetricIouJaccard(MetricPerSampleDefault):
    """
    Compute IOU Jaccard score for every label
    used for sematric and binary segmentation
    """

    def __init__(
        self,
        pred: str,
        target: str,
        pixel_weight: Optional[str] = None,
        class_weights: Optional[Dict[int, float]] = None,
        **kwargs
    ):
        """
        See super class for the missing params
        :param class_weights: weight per segmentation class , we assume sum of total weights is 1 and each element is in 0-1 range
        :param pixel_weight: Optional dictionary key to collect
        """
        average = partial(average_sample_results, class_weights=class_weights)
        super().__init__(
            pred,
            target,
            pixel_weight=pixel_weight,
            metric_per_sample_func=MetricsSegmentation.iou_jaccard,
            result_aggregate_func=average,
            **kwargs
        )


class MetricOverlap(MetricPerSampleDefault):
    """
    Compute overlap score for every label
    used for sematric and binary segmentation
    """

    def __init__(
        self,
        pred: str,
        target: str,
        pixel_weight: Optional[str] = None,
        class_weights: Optional[Dict[int, float]] = None,
        **kwargs
    ):
        """
        See super class for the missing params
        :param class_weights: weight per segmentation class , we assume sum of total weights is 1 and each element is in 0-1 range
        :param pixel_weight: Optional dictionary key to collect
        """
        average = partial(average_sample_results, class_weights=class_weights)
        super().__init__(
            pred,
            target,
            pixel_weight=pixel_weight,
            metric_per_sample_func=MetricsSegmentation.overlap,
            result_aggregate_func=average,
            **kwargs
        )


class Metric2DHausdorff(MetricPerSampleDefault):
    """
    Compute Hausdorff score for every label - works for 2D array only!
    used for sematric and binary segmentation
    """

    def __init__(self, pred: str, target: str, class_weights: Optional[Dict[int, float]] = None, **kwargs):
        """
        See super class for the missing params
        :param class_weights: weight per segmentation class , we assume sum of total weights is 1 and each element is in 0-1 range
        """
        average = partial(average_sample_results, class_weights=class_weights)
        super().__init__(
            pred,
            target,
            metric_per_sample_func=MetricsSegmentation.hausdorff_2d_distance,
            result_aggregate_func=average,
            **kwargs
        )


class MetricPixelAccuracy(MetricPerSampleDefault):
    """
    Compute pixel accuracy score for every label
    used for sematric and binary segmentation
    """

    def __init__(
        self,
        pred: str,
        target: str,
        pixel_weight: Optional[str] = None,
        class_weights: Optional[Dict[int, float]] = None,
        **kwargs
    ):
        """
        See super class for the missing params
        :param class_weights: weight per segmentation class , we assume sum of total weights is 1 and each element is in 0-1 range
        :param pixel_weight: Optional dictionary key to collect
        """
        average = partial(average_sample_results, class_weights=class_weights)
        super().__init__(
            pred,
            target,
            pixel_weight=pixel_weight,
            metric_per_sample_func=MetricsSegmentation.pixel_accuracy,
            result_aggregate_func=average,
            **kwargs
        )
