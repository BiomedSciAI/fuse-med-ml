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

from fuse.eval.metrics.classification.metrics_classification_common import MetricMultiClassDefault
from fuse.eval.metrics.libs.survival import MetricsSurvival


class MetricCIndex(MetricMultiClassDefault):
    """
    Compute auc roc (Receiver operating characteristic) score using sklearn (one vs rest)
    """

    def __init__(
        self,
        pred: str,
        target: str,
        class_names: Optional[Sequence[str]] = None,
        event_observed : Optional[Sequence[np.ndarray]] = None,
        **kwargs,
    ):
        """
        See MetricMultiClassDefault for the missing params
        :param max_fpr: float > 0 and <= 1, default=None
                        If not ``None``, the standardized partial AUC over the range [0, max_fpr] is returned.
        """
        c_index = partial(MetricsSurvival.c_index, event_observed=event_observed)
        super().__init__(pred, target, metric_func=c_index, class_names=class_names, **kwargs)
