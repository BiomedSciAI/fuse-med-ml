import numpy as np
from typing import Sequence, Union


class Stat:
    """
    Statistical metrics
    """

    @staticmethod
    def pearson_correlation(
        pred: Union[np.ndarray, Sequence], target: Union[np.ndarray, Sequence]
    ) -> float:
        """
        Pearson correlation coefficient measuring the linear relationship between two datasets/vectors.
        """
        if isinstance(pred, Sequence):
            pred = np.array(pred)
        if isinstance(target, Sequence):
            target = np.array(target)
        pred = pred.squeeze()
        target = target.squeeze()
        if len(pred.shape) > 1 or len(target.shape) > 1:
            raise ValueError(
                f"expected 1D vectors. got pred shape: {pred.shape}, target shape: {target.shape}"
            )

        mean_pred = np.mean(pred)
        mean_target = np.mean(target)

        r = np.sum((pred - mean_pred) * (target - mean_target)) / np.sqrt(
            np.sum((pred - mean_pred) ** 2) * np.sum((target - mean_target) ** 2)
        )

        return r
