import numpy as np
from typing import Sequence, Union
from scipy.stats import pearsonr, spearmanr


class Stat:
    """
    Statistical metrics
    """

    @staticmethod
    def pearson_correlation(
        pred: Union[np.ndarray, Sequence],
        target: Union[np.ndarray, Sequence],
        mask: Union[np.ndarray, Sequence, None] = None,
    ) -> dict:
        """
        Pearson correlation coefficient measuring the linear relationship between two datasets/vectors.
        :param pred: prediction values
        :param target: target values
        :param mask: optional boolean mask. if it is provided, the metric will be applied only to the masked samples
        """
        if 0 == len(pred):
            return dict(statistic=float("nan"), p_value=float("nan"))

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
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        pred = pred.squeeze()
        target = target.squeeze()
        if len(pred.shape) > 1 or len(target.shape) > 1:
            raise ValueError(
                f"expected 1D vectors. got pred shape: {pred.shape}, target shape: {target.shape}"
            )

        statistic, p_value = pearsonr(pred, target)

        results = {}
        results["statistic"] = statistic
        results["p_value"] = p_value
        return results

    @staticmethod
    def spearman_correlation(
        pred: Union[np.ndarray, Sequence],
        target: Union[np.ndarray, Sequence],
        mask: Union[np.ndarray, Sequence, None] = None,
    ) -> dict:
        """
        Spearman correlation coefficient measuring the monotonic relationship between two datasets/vectors.
        :param pred: prediction values
        :param target: target values
        :param mask: optional boolean mask. if it is provided, the metric will be applied only to the masked samples
        """
        if 0 == len(pred):
            return dict(statistic=float("nan"), p_value=float("nan"))

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
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        pred = pred.squeeze()
        target = target.squeeze()
        if len(pred.shape) > 1 or len(target.shape) > 1:
            raise ValueError(
                f"expected 1D vectors. got pred shape: {pred.shape}, target shape: {target.shape}"
            )

        assert len(pred) == len(
            target
        ), f"Spearman corr expected to get pred and target with same length but got pred={len(pred)} - target={len(target)}"

        statistic, p_value = spearmanr(
            pred, target, nan_policy="propagate"
        )  # nans will result in nan outputs

        results = {}
        results["statistic"] = statistic
        results["p_value"] = p_value
        return results
