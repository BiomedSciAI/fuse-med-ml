from functools import partial
from typing import List, Optional
from fuse.eval.metrics.libs.instance_segmentation import MetricsInstanceSegmentaion

import numpy as np

from fuse.eval.metrics.metrics_common import MetricPerSampleDefault


def calculate_precision(metric_result: List[float], threshold: float = 0.5) -> float:
    """
    Calculates average precision under threshold for detection
    :param metric_result:  matrix of similarity score between prediction and target , (axis 0 is number of annotations, axis 1 is number of targets)
    :param threshold: a number which determines the metric value of which above we consider for detection
                      it's purpouse to ignore misdetected instances , 0.5 is the common threshold on iou metric
    :return: average result over images
    """
    per_instane_results = []
    for res in metric_result:
        metric_maximum = res.max(axis=0)
        metric_maximum = metric_maximum[metric_maximum > threshold]
        precision = float(len(metric_maximum)) / float(res.shape[0])
        per_instane_results.append(precision)
    result = np.mean(per_instane_results)
    return result


def calculate_recall(metric_result: List[float], threshold: float = 0.5) -> float:
    """
    Calculates average recall under threshold for detection
    :param metric_result: matrix of similarity score between prediction and target , (axis 0 is number of annotations, axis 1 is number of targets)
    :param threshold: a number which determines the metric value of which above we consider for detection
                      it's purpouse to ignore misdetected instances , 0.5 is the common threshold on iou metric
    :return: average result over images
    """
    per_instane_results = []
    for res in metric_result:
        metric_maximum = res.max(axis=0)
        metric_maximum = metric_maximum[metric_maximum > threshold]
        recall = float(len(metric_maximum)) / float(res.shape[1])
        per_instane_results.append(recall)
    result = np.mean(per_instane_results)
    return result


class MetricDetectionPrecision(MetricPerSampleDefault):
    """
    Compute detection precision based on IOU Jaccard score for binary instance segmentation input
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
        See super class for the missing params, to read more about the segmentation types go to class MetricsInstanceSegmentaion
        :param segmentation_pred_type: input pred format - pixel_map , uncompressed_RLE , compressed_RLE , polygon , bbox
        :param segmentation_target_type: input target format - pixel_map , uncompressed_RLE , compressed_RLE , polygon , bbox
        :param height: height of the original image ( y axis)
        :param width: width of the original image ( x axis)
        :param threshold: a number which determines the metric value of which above we consider for detection
                          it's purpouse to ignore misdetected instances , 0.5 is the common threshold on iou metric
        """
        iou = partial(
            MetricsInstanceSegmentaion.iou_jaccard,
            segmentation_pred_type=segmentation_pred_type,
            segmentation_target_type=segmentation_target_type,
        )
        precision = partial(calculate_precision, threshold=threshold)
        super().__init__(
            pred=pred,
            target=target,
            height=height,
            width=width,
            metric_per_sample_func=iou,
            result_aggregate_func=precision,
            **kwargs
        )


class MetricDetectionRecall(MetricPerSampleDefault):
    """
    Compute detection recall based on IOU Jaccard score for binary instance segmentation input
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
        See super class for the missing params, to read more about the segmentation types go to class MetricsInstanceSegmentaion
        :param segmentation_pred_type: input pred format - pixel_map , uncompressed_RLE , compressed_RLE , polygon , bbox
        :param segmentation_target_type: input target format - pixel_map , uncompressed_RLE , compressed_RLE , polygon , bbox
        :param height: height of the original image ( y axis)
        :param width: width of the original image ( x axis)
        :param threshold: a number which determines the metric value of which above we consider for detection
                          it's purpouse to ignore misdetected instances , 0.5 is the common threshold on iou metric
        """
        iou = partial(
            MetricsInstanceSegmentaion.iou_jaccard,
            segmentation_pred_type=segmentation_pred_type,
            segmentation_target_type=segmentation_target_type,
        )
        recall = partial(calculate_recall, threshold=threshold)
        super().__init__(
            pred=pred,
            target=target,
            height=height,
            width=width,
            metric_per_sample_func=iou,
            result_aggregate_func=recall,
            **kwargs
        )
