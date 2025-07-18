from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from fuse.eval.metrics.libs.stat import Stat
from fuse.eval.metrics.metrics_common import MetricDefault


class MetricPearsonCorrelation(MetricDefault):
    def __init__(
        self, pred: str, target: str, mask: str | None = None, **kwargs: dict
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
        self, pred: str, target: str, mask: str | None = None, **kwargs: dict
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
        mask: str | None = None,
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
            mask=mask,
            metric_func=self.mae,
            **kwargs,
        )

    def mae(
        self,
        pred: list | np.ndarray,
        target: list | np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs: dict,
    ) -> float:
        if mask is not None:
            if isinstance(pred, Sequence):
                if np.isscalar(pred[0]):
                    pred = np.array(pred)
                else:
                    pred = np.concatenate(pred)
            if isinstance(target, Sequence):
                if np.isscalar(target[0]):
                    target = np.array(target)
                else:
                    target = np.concatenate(target)
            if isinstance(mask, Sequence):
                if np.isscalar(mask[0]):
                    mask = np.array(mask).astype("bool")
                else:
                    mask = np.concatenate(mask).astype("bool")

            target = target[mask]
            pred = pred[mask]

        return mean_absolute_error(y_true=target, y_pred=pred)


class MetricMSE(MetricDefault):
    def __init__(
        self,
        pred: str,
        target: str,
        mask: str | None = None,
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
            mask=mask,
            metric_func=self.mse,
            **kwargs,
        )

    def mse(
        self,
        pred: list | np.ndarray,
        target: list | np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs: dict,
    ) -> float:
        if mask is not None:
            if isinstance(pred, Sequence):
                if np.isscalar(pred[0]):
                    pred = np.array(pred)
                else:
                    pred = np.concatenate(pred)
            if isinstance(target, Sequence):
                if np.isscalar(target[0]):
                    target = np.array(target)
                else:
                    target = np.concatenate(target)
            if isinstance(mask, Sequence):
                if np.isscalar(mask[0]):
                    mask = np.array(mask).astype("bool")
                else:
                    mask = np.concatenate(mask).astype("bool")

            target = target[mask]
            pred = pred[mask]

        return mean_squared_error(y_true=target, y_pred=pred)


class MetricRMSE(MetricDefault):
    def __init__(
        self,
        pred: str,
        target: str,
        mask: str | None = None,
        **kwargs: dict,
    ) -> None:
        """
        See MetricDefault for the missing params
        :param pred: scalar predictions
        :param target: ground truth scalar labels
        """
        super().__init__(
            pred=pred,
            target=target,
            mask=mask,
            metric_func=self.rmse,
            **kwargs,
        )

    def rmse(
        self,
        pred: list | np.ndarray,
        target: list | np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs: dict,
    ) -> float:
        if mask is not None:
            if isinstance(pred, Sequence):
                if np.isscalar(pred[0]):
                    pred = np.array(pred)
                else:
                    pred = np.concatenate(pred)
            if isinstance(target, Sequence):
                if np.isscalar(target[0]):
                    target = np.array(target)
                else:
                    target = np.concatenate(target)
            if isinstance(mask, Sequence):
                if np.isscalar(mask[0]):
                    mask = np.array(mask).astype("bool")
                else:
                    mask = np.concatenate(mask).astype("bool")

            target = target[mask]
            pred = pred[mask]

        pred = np.array(pred).flatten()
        target = np.array(target).flatten()

        assert len(pred) == len(
            target
        ), f"Expected pred and target to have the dimensions but found: {len(pred)} elements in pred and {len(target)} in target"

        return root_mean_squared_error(y_true=target, y_pred=pred)


class MetricR2(MetricDefault):
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
        """
        super().__init__(
            pred=pred,
            target=target,
            metric_func=self.r2,
            **kwargs,
        )

    def r2(
        self,
        pred: list | np.ndarray,
        target: list | np.ndarray,
        **kwargs: dict,
    ) -> float:
        pred = np.array(pred).flatten()
        target = np.array(target).flatten()

        assert len(pred) == len(
            target
        ), f"Expected pred and target to have the dimensions but found: {len(pred)} elements in pred and {len(target)} in target"

        return r2_score(target, pred)
