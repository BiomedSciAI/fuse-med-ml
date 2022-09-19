from typing import Optional, Sequence, Hashable
from fuse.eval.metrics.libs.ensembling import Ensembling
from fuse.eval.metrics.utils import PerSampleData

import pandas as pd

import numpy as np

from fuse.eval.metrics.metrics_common import MetricDefault
from functools import partial


class MetricEnsemble(MetricDefault):
    def __init__(
        self,
        pred_keys: Sequence[str],
        target: Optional[str] = None,
        method: Optional[str] = "average",
        output_file: Optional[str] = None,
        output_pred_key: Optional[str] = "preds",
        output_target_key: Optional[str] = "target",
        **kwargs,
    ):
        """
        Model ensembling metric.
        It obtains as input predictions from multiple models on the same set of samples
        and outputs a refined prediction using average, median or voting.
        :param pred_keys: List of key names (strings) for the multiple model prediction scores
        :param target: Optional key for target values. Only needed for optional saving the targets in an output file.
        :param method: Ensembling method. 'average' or 'voting'.
            if 'average', the predictions are assumed to be continuous (probabilities or regression output)
            if 'voting', the predictions are assumed to be class predictions (integers)
        :param output_file: Optional output filename
        :param output_pred_key: output key name for the predictions
        :param output_target_key: output key name for the target
        """
        ensemble = partial(
            self._ensemble,
            method=method,
            output_file=output_file,
            output_pred_key=output_pred_key,
            output_target_key=output_target_key,
        )
        for i, key in enumerate(pred_keys):
            kwargs["pred" + str(i)] = key
        super().__init__(metric_func=ensemble, target=target, extract_ids=True, **kwargs)

    def _ensemble(
        self,
        ids: Sequence[Hashable],
        target: Optional[Sequence[np.ndarray]] = None,
        method: Optional[str] = "average",
        output_file: Optional[str] = None,
        output_pred_key: Optional[str] = "preds",
        output_target_key: Optional[str] = "target",
        **kwargs,
    ):
        preds = [kwargs[k] for k in kwargs if k.startswith("pred")]
        preds = list(np.stack(preds, axis=1))
        ensemble_preds = Ensembling.ensemble(preds=preds, method=method)
        # make sure to return the per-sample metric result for the relevant sample ids:
        per_sample_data = PerSampleData(data=ensemble_preds, ids=ids)
        if target is not None:
            per_sample_target = PerSampleData(data=target, ids=ids)
        if output_file is not None:
            df = pd.DataFrame()
            df["id"] = ids
            df[output_pred_key] = per_sample_data(ids)
            if target is not None:
                df[output_target_key] = per_sample_target(ids)
            df.to_pickle(output_file, compression="gzip")
        return {output_pred_key: per_sample_data, output_target_key: target}
