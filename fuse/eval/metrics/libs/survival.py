from typing import Any, Dict, Optional, Sequence, Tuple, Union
from lifelines.utils import concordance_index
import numpy as np

class MetricsSurvival:
    @staticmethod
    def c_index(pred: np.ndarray, target: np.ndarray = None, event_observed : np.ndarray = None ) -> float:
        """
        Compute c-index (concordance index) score using lifelines
        :param pred: prediction array per sample. Each element shape [num_classes]
        :param target: target per sample. Each element is an integer in range [0 - num_classes)
        :param event_observed: Optioal- a length-n iterable censoring flags, 1 if observed, 0 if not. Default None assumes all observed.
        :return c-index (concordance index) score
        """
        return concordance_index(target, pred, event_observed)