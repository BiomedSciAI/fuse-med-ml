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
from typing import Any, Callable, Dict, Optional, Sequence, Union, Hashable

import numpy as np
import pandas as pd

from fuse.eval.metrics.metrics_common import MetricBase, MetricWithCollectorBase
from fuse.eval.metrics.libs.model_comparison import ModelComparison


class PairedBootstrap(MetricWithCollectorBase):
    """
    Verify your hypothesis about the relation between models using bootstrap approach
    The specific hypothesis can be specified in compare_method which by default checks superiority.
    Typically, the compare_method will compute p_value:
    the probability of obtaining test results at least as extreme as the results actually observed,
    under the assumption that the null hypothesis is correct.
    The format of the result dictionary returned by eval() method may varied and depends in comapre_method used
    """

    def __init__(
        self,
        metric_test: MetricBase,
        metric_reference: MetricBase,
        stratum: str,
        metric_keys_to_compare: Optional[Sequence[str]] = None,
        compare_method: Callable = ModelComparison.bootstrap_margin_superiority,
        num_of_bootstraps: int = 10000,
        rnd_seed: int = 1234,
        margin: float = 0.0,
        **super_kwargs
    ) -> None:
        """
        :param metric_test: compare the results of metric_test to results of metric_reference
        :param metric_reference: compare the results of metric_test to results of metric_reference
        :param stratum: if the resampling should be done by strata, specify the key in batch_dict to collect data from
        :param num_of_bootstraps: number of bootstrap iterations
        :param rnd_seed: seed for random number generator.
        :param margin: the difference required to be considered significant by the compare_method
        :param metrics_keys_to_compare: compare the following metrics values: keys in the dictionary returned by metric_test and metric_reference.
                                        Set to None if those metrics returns single value.
        :param compare_method: callback that defines the hypothesis and compares the results.
                               Required prototype: func(name: str, test_values: Sequence[float], reference_values: Sequence[float]) -> Union[Dict[str, float], float]
                               See example: ModelComparison.bootstrap_margin_superiority , ModelComparison.bootstrap_margin_non_inferiority , ModelComparison.bootstrap_margin_equality
        """

        super().__init__(stratum=stratum, **super_kwargs)

        # store arguments
        self._metric_test = metric_test
        self._metric_reference = metric_reference
        self._num_of_bootstraps = num_of_bootstraps
        self._rnd_seed = rnd_seed
        self._metric_keys_to_compare = metric_keys_to_compare
        self._compare_method = compare_method
        self._margin = margin

    def collect(self, batch: Dict) -> None:
        "See super class"
        self._metric_test.collect(batch)
        self._metric_reference.collect(batch)
        return super().collect(batch)

    def set(self, data: Union[Dict, Sequence[Dict], pd.DataFrame]) -> None:
        "See super class"
        self._metric_test.set(data)
        self._metric_reference.set(data)
        return super().set(data)

    def reset(self) -> None:
        "See super class"
        self._metric_test.reset()
        self._metric_reference.reset()
        return super().reset()

    def eval(self, results: Dict[str, Any] = None, ids: Sequence[Hashable] = None) -> Dict[str, Any]:
        """
        See super class
        :return: the compared results dictionary created by compare_method() specified in constructor
        """
        # prepare data
        data = self._collector.get(ids)
        if ids is None:
            ids = self._collector.get_ids()
        if not ids:
            raise Exception(
                "Error: paired bootstrap is supported only when a unique identifier is specified. Add key 'id' to your data"
            )
        ids = np.array(ids)

        rnd = np.random.RandomState(self._rnd_seed)
        stratum_id = np.array(data["stratum"]) if "stratum" in data else np.ones(len(data["id"]))
        unique_strata = np.unique(stratum_id)

        # initialize results
        if self._metric_keys_to_compare is None:
            bootstrap_results_test = np.empty(self._num_of_bootstraps)
            bootstrap_results_reference = np.empty(self._num_of_bootstraps)
        else:
            bootstrap_results_test = {key: np.empty(self._num_of_bootstraps) for key in self._metric_keys_to_compare}
            bootstrap_results_reference = {
                key: np.empty(self._num_of_bootstraps) for key in self._metric_keys_to_compare
            }

        # aggregate bootstrap results
        for bs_index in range(self._num_of_bootstraps):
            sampled_ids = ids.copy()
            for stratum in unique_strata:
                stratum_filter = stratum_id == stratum
                n_stratum = sum(stratum_filter)
                random_sample = rnd.randint(0, n_stratum, size=n_stratum)
                sampled_ids[stratum_filter] = ids[stratum_filter][random_sample]
            result_test = self._metric_test.eval(results, sampled_ids)
            result_reference = self._metric_reference.eval(results, sampled_ids)

            # single-value metric case
            if self._metric_keys_to_compare is None:
                assert not isinstance(
                    result_test, Dict
                ), "Error: metric_test returned a dictionary of results. Specify the relevant keys to compare in metric_keys_to_compare."
                assert not isinstance(
                    result_reference, Dict
                ), "Error: metric_reference returned a dictionary of results. Specify the relevant keys to compare in metric_keys_to_compare."
                bootstrap_results_test[bs_index] = result_test
                bootstrap_results_reference[bs_index] = result_reference

            # multi-values metric case
            if self._metric_keys_to_compare is not None:
                assert isinstance(
                    result_test, Dict
                ), "Error: metric_test returned a single value. Set metric_keys_to_compare to None."
                assert isinstance(
                    result_reference, Dict
                ), "Error: metric_reference returned a single value. Set metric_keys_to_compare to None."
                for key in self._metric_keys_to_compare:
                    bootstrap_results_test[key][bs_index] = result_test[key]
                    bootstrap_results_reference[key][bs_index] = result_reference[key]

        # compare
        if self._metric_keys_to_compare is None:
            return self._compare_method(None, bootstrap_results_test, bootstrap_results_reference, margin=self._margin)

        metric_results = {}
        for key in bootstrap_results_test:
            metric_results[key] = self._compare_method(
                key, bootstrap_results_test[key], bootstrap_results_reference[key], margin=self._margin
            )
        return metric_results
