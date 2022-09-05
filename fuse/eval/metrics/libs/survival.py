from typing import Any, Dict, Optional, Sequence, Tuple, Union
from lifelines.utils import concordance_index
import numpy as np

class MetricsSurvival:
    @staticmethod
    def c_index(pred: np.ndarray, target: np.ndarray = None, event_observed : np.ndarray = None ) -> float:
        return concordance_index(pred,target,event_observed)