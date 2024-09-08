from typing import Any, Dict, Hashable, Optional, Sequence, Union, List
from collections import Counter
from fuse.eval.metrics.metrics_common import MetricDefault, MetricWithCollectorBase
from fuse.eval.metrics.libs.stat import Stat

import numpy as np  # is this import an issue here? if so can move to other dir
from sklearn.metrics import mean_absolute_error, mean_squared_error


class MetricUniqueValues(MetricWithCollectorBase):
    """
    Collect the all the categorical values and the number of occurrences
    Result format: list of tuples - each tuple include the value and number of occurrences
    """

    def __init__(self, key: str, **kwargs: dict) -> None:
        super().__init__(key=key, **kwargs)

    def eval(
        self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None
    ) -> None:
        values = self._extract_arguments(results, ids)["key"]
        counter = Counter(values)

        return list(counter.items())


class MetricPearsonCorrelation(MetricDefault):
    def __init__(
        self, pred: str, target: str, mask: Optional[str] = None, **kwargs: dict
    ) -> None:
        super().__init__(
            pred=pred,
            target=target,
            mask=mask,
            metric_func=Stat.pearson_correlation,
            **kwargs,
        )


class MetricSpearmanCorrelation(MetricDefault):
    def __init__(
        self, pred: str, target: str, mask: Optional[str] = None, **kwargs: dict
    ) -> None:
        super().__init__(
            pred=pred,
            target=target,
            mask=mask,
            metric_func=Stat.spearman_correlation,
            **kwargs,
        )


class MetricMAE(MetricDefault):
    def __init__(
        self,
        pred: str,
        target: str,
        **kwargs: dict,
    ) -> None:
        """
        See MetricDefault for the missing params
        :param pred: scalar predictions
        :param target: ground truth scalar labels
        :param threshold: threshold to apply to both pred and target
        :param balanced: optionally to use balanced accuracy (from sklearn) instead of regular accuracy.
        """
        super().__init__(
            pred=pred,
            target=target,
            metric_func=self.mae,
            **kwargs,
        )

    def mae(
        self,
        pred: Union[List, np.ndarray],
        target: Union[List, np.ndarray],
        **kwargs: dict,
    ) -> float:
        return mean_absolute_error(y_true=target, y_pred=pred)


class MetricMSE(MetricDefault):
    def __init__(
        self,
        pred: str,
        target: str,
        **kwargs: dict,
    ) -> None:
        """
        Our implementation of standard MSE, current version of scikit dones't support it as a metric.
        See MetricDefault for the missing params
        :param pred: scalar predictions
        :param target: ground truth scalar labels
        :param threshold: threshold to apply to both pred and target
        :param balanced: optionally to use balanced accuracy (from sklearn) instead of regular accuracy.
        """
        super().__init__(
            pred=pred,
            target=target,
            metric_func=self.mse,
            **kwargs,
        )

    def mse(
        self,
        pred: Union[List, np.ndarray],
        target: Union[List, np.ndarray],
        **kwargs: dict,
    ) -> float:
        return mean_squared_error(y_true=target, y_pred=pred)


class MetricRMSE(MetricDefault):
    def __init__(
        self,
        pred: str,
        target: str,
        **kwargs: dict,
    ) -> None:
        """
        See MetricDefault for the missing params
        :param pred: scalar predictions
        :param target: ground truth scalar labels
        :param threshold: threshold to apply to both pred and target
        :param balanced: optionally to use balanced accuracy (from sklearn) instead of regular accuracy.
        """
        super().__init__(
            pred=pred,
            target=target,
            metric_func=self.mse,
            **kwargs,
        )

    def mse(
        self,
        pred: Union[List, np.ndarray],
        target: Union[List, np.ndarray],
        **kwargs: dict,
    ) -> float:

        pred = np.array(pred).flatten()
        target = np.array(target).flatten()

        assert len(pred) == len(
            target
        ), f"Expected pred and target to have the dimensions but found: {len(pred)} elements in pred and {len(target)} in target"

        squared_diff = (pred - target) ** 2
        return squared_diff.mean()
