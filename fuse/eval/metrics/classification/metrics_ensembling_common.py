
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, Hashable
from fuse.eval.metrics.libs.ensembling import Ensembling
from fuse.eval.metrics.utils import PerSampleData

import pandas as pd
import traceback

import numpy as np

from fuse.eval.metrics.metrics_common import MetricWithCollectorBase
from .metrics_classification_common import MetricMultiClassDefault

class  MetricEnsemble(MetricWithCollectorBase):
    def __init__(self, preds: str, **kwargs):
        """
        :param preds: key name for the multiple model prediction scores
        """
        super().__init__(pred=preds, target=None, extract_ids=True, metric_func=self._ensemble, **kwargs)

    def _ensemble(self, pred: Sequence[np.ndarray], ids: Sequence[Hashable]):

        ensemble_preds = Ensembling.ensemble(pred=preds)
        # make sure to return the per-sample metric result for the relevant sample ids:
        per_sample_data = PerSampleData(data=ensemble_preds, ids=ids)

        return {'preds': per_sample_data}