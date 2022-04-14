
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, Hashable
from fuse.eval.metrics.libs.ensembling import Ensembling
from fuse.eval.metrics.utils import PerSampleData

import pandas as pd
import traceback

import numpy as np

from fuse.eval.metrics.metrics_common import MetricDefault
from .metrics_classification_common import MetricMultiClassDefault
from functools import partial

class  MetricEnsemble(MetricDefault):
    def __init__(self, preds: str, output_file: Optional[str]=None, **kwargs):
        """
        :param preds: key name for the multiple model prediction scores
        """
        ensemble = partial(self._ensemble, output_file=output_file)
        super().__init__(metric_func=ensemble, preds=preds, extract_ids=True, **kwargs)

    def _ensemble(self, preds: Sequence[np.ndarray], ids: Sequence[Hashable], output_file: Optional[str]=None):

        ensemble_preds = Ensembling.ensemble(preds=preds)
        # make sure to return the per-sample metric result for the relevant sample ids:
        per_sample_data = PerSampleData(data=ensemble_preds, ids=ids)

        if output_file is not None:
            df = pd.DataFrame()
            df['id'] = ids
            df['preds'] = per_sample_data(ids)
            df.to_pickle(output_file, compression='gzip')
        return {'preds_ensembled': per_sample_data}