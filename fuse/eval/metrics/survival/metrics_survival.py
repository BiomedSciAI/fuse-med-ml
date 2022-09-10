"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Hashable
import traceback

import numpy as np

from fuse.eval.metrics.libs.survival import MetricsSurvival
from fuse.eval.metrics.metrics_common import MetricWithCollectorBase



class MetricCIndex(MetricWithCollectorBase):
    """
    Compute c-index (concordance index) score using lifelines
    """

    def __init__(
        self,
        pred_key: str,
        event_times_key: str,
        event_observed_key: str,
        **kwargs,
    ):
        """
        See MetricMultiClassDefault for the missing params
        :param target:  a length-n iterable censoring flags, 1 if observed, 0 if not. Default None assumes all observed.
        """
        super().__init__(pred_key=pred_key, event_times_key=event_times_key, event_observed_key=event_observed_key, **kwargs)


    def eval(
        self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None
    ) -> Union[Dict[str, Any], Any]:
        """
        See super class
        """
        # extract values from collected data and results dict
        kwargs = self._extract_arguments(results, ids)

       
        # single evaluation for all classes at once / binary classifier
        try:
            metric_results =MetricsSurvival.c_index(predicted_scores=np.asarray(kwargs['pred_key'])[:,1], 
                                                    event_times=np.asarray(kwargs['event_times_key']), 
                                                    event_observed = np.asarray(kwargs['event_observed_key']))
        except:
            track = traceback.format_exc()
            print(f"Error in metric: {track}")
            metric_results = None
        return metric_results
