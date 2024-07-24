import numpy as np
from typing import Sequence, Union
from scipy.stats import pearsonr


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
        if isinstance(pred, Sequence):
            pred = np.concatenate(pred)
        if isinstance(target, Sequence):
            target = np.concatenate(target)
        if isinstance(mask, Sequence):
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
