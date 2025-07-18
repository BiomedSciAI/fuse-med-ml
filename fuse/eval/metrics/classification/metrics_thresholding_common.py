from collections.abc import Hashable, Sequence
from typing import Any, Dict, Tuple

import numpy as np

from fuse.eval.metrics.libs.thresholding import Thresholding
from fuse.eval.metrics.utils import PerSampleData

from .metrics_classification_common import MetricMultiClassDefault


class MetricApplyThresholds(MetricMultiClassDefault):
    def __init__(
        self,
        pred: str,
        class_names: Sequence[str] | None = None,
        operation_point: float | Sequence[Tuple[int, float]] | str | None = None,
        **kwargs: dict
    ):
        """
        :param pred: key name for the model prediction scores
        :param class_names: class names. required for multi-class classifiers
        :param operation_point: Optional. If specified will be used to convert probabilities to class prediction.
                                          Options:
                                          * None - argmax(probability) - default mode
                                          * float - used as a threshold for the positive class in binary classification
                                          * [(<class_index_first>, <threshold_first>), ...,(<class_index_last>, <threshold_last>)] -
                                            the class prediction will be the first class in which the probability cross the threshold.
                                            If no class probability crossed the threshold the predicated class will be set to -1
        """
        super().__init__(
            pred=pred,
            target=None,
            extract_ids=True,
            metric_func=self._apply_thresholds,
            class_names=class_names,
            operation_point=operation_point,
            **kwargs
        )

    def _apply_thresholds(
        self,
        pred: Sequence[np.ndarray],
        ids: Sequence[Hashable],
        operation_point: float | Sequence[Tuple[int, float]] | str | None = None,
    ) -> Dict[str, Any]:
        pred_thresholded = Thresholding.apply_thresholds(
            pred=pred, operation_point=operation_point
        )
        # make sure to return the per-sample metric result for the relevant sample ids:
        per_sample_data = PerSampleData(data=pred_thresholded, ids=ids)

        return {"cls_pred": per_sample_data}
