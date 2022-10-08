from typing import Any, Dict, Optional, Sequence, Tuple, Union
from lifelines.utils import concordance_index
import numpy as np
from copy import copy
class MetricsSurvival:
    @staticmethod
    def c_index(pred: np.ndarray, event_times: np.ndarray = None, event_observed : np.ndarray = None, event_class_index: int = -1, time_unit:int=1) -> float:
        """
        Compute c-index (concordance index) score using lifelines
        :param pred: prediction array per sample. Each element is a score (scalar). Higher score - higher chance for the event
        :param event_times: event/censor time array
        :param event_observed: Optioal- a length-n iterable censoring flags, 1 if observed, 0 if not. Default None assumes all observed.
        :param event_class_index: Optional - the index of the event class in predicted scores tuples
        :return c-index (concordance index) score
        """


        if isinstance(pred[0], np.ndarray):
            pred = np.asarray(pred)[:,event_class_index]

        if time_unit > 1:
            event_times = copy(event_times)
            for i in range(len(event_times)):
                event_times[i]  = int(event_times[i] / time_unit)
           
        return concordance_index(event_times, -pred, event_observed) 