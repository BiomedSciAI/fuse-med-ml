from typing import Any, Dict, Optional, Sequence, Tuple, Union
from lifelines.utils import concordance_index
import numpy as np

class MetricsSurvival:
    @staticmethod
    def c_index(predicted_scores: np.ndarray, event_times: np.ndarray = None, event_observed : np.ndarray = None ) -> float:
        """
        Compute c-index (concordance index) score using lifelines
        :param predicted_scores: prediction array per sample. Each element is a score (scalar)
        :param event_times: event/censor time array
        :param event_observed: Optioal- a length-n iterable censoring flags, 1 if observed, 0 if not. Default None assumes all observed.
        :return c-index (concordance index) score
        """
        return concordance_index(event_times, predicted_scores, event_observed) 