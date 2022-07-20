from functools import partial
from typing import List, Optional
from fuse.eval.metrics.libs.instance_segmentation import MetricsInstanceSegmentaion

import numpy as np

from fuse.eval.metrics.metrics_common import MetricPerSampleDefault


def average_sample_results(metric_result: List[float], threshold: float = 0.5) -> float:
    """
    Calculates average result per class and average result over classes on a specific metric
    metric_result assumed to have same type of keys as in class_weights which represents the different classes
    :param metric_result: list of per image metric results ,each element is a dictionary where the key is class id and value is the metric score
    :param threshold: a number which determines the metric value of which above we consider for averaging
                      it's purpouse to ignore misdetected instances , 0.5 is the common threshold on iou metric
    :return: average result over images
    """
    per_instane_results = []
    for res in metric_result:
        metric_maximum = res.max(axis=0)
        metric_maximum = metric_maximum[metric_maximum > threshold]
        if len(metric_maximum) > 0:
            per_instane_results.append(metric_maximum.mean())
    result = np.mean(per_instane_results)
    return result


class MetricInstanceIouJaccard(MetricPerSampleDefault):
    """
    Compute aggregated IOU Jaccard score for binary instance segmentation input
    """

    def __init__(
        self,
        pred: str,
        target: str,
        segmentation_pred_type: str,
        segmentation_target_type: str,
        height: str,
        width: str,
        threshold: Optional[float] = 0.5,
        **kwargs
    ):
        """
        See super class for the missing params , to read more about the segmentation types go to class MetricsInstanceSegmentaion
        :param segmentation_pred_type: input pred format - pixel_map , uncompressed_RLE , compressed_RLE , polygon , bbox
        :param segmentation_target_type: input target format - pixel_map , uncompressed_RLE , compressed_RLE , polygon , bbox
        :param height: height of the original image ( y axis)
        :param width: width of the original image ( x axis)
        :param threshold: a number which determines the metric value of which above we consider for averaging
                          it's purpouse to ignore misdetected instances , 0.5 is the common threshold on iou metric
        """
        iou = partial(
            MetricsInstanceSegmentaion.iou_jaccard,
            segmentation_pred_type=segmentation_pred_type,
            segmentation_target_type=segmentation_target_type,
        )
        average = partial(average_sample_results, threshold=threshold)
        super().__init__(
            pred=pred,
            target=target,
            height=height,
            width=width,
            metric_per_sample_func=iou,
            result_aggregate_func=average,
            **kwargs
        )
