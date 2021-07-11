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

from copy import deepcopy
from enum import Enum

import numpy as np
import scipy.stats
from typing import Dict, Tuple, Optional

from fuse.metrics.metric_base import FuseMetricBase


class CIMethod(Enum):
    """
    Method to compute confidence interval from a list of bootstrap results
    """
    NORMAL = 0
    PIVOTAL = 1
    PERCENTILE = 2

    def compute_confidence_interval(self, org_statistic: np.ndarray, bootstrap_statistics: np.ndarray, confidence: float) -> Tuple[float, float]:
        lower_confidence_ratio = (1 - confidence / 100.0) / 2.0
        if self == CIMethod.NORMAL:
            sigma = np.std(bootstrap_statistics)
            c_alpha = scipy.stats.norm.isf(lower_confidence_ratio)
            return org_statistic - c_alpha * sigma, org_statistic + c_alpha * sigma
        statistic_observations = np.sort(bootstrap_statistics)

        n = len(statistic_observations)
        index_low = int(np.floor(lower_confidence_ratio * n))
        index_high = int(np.floor((1 - lower_confidence_ratio) * n))

        bootstrap_statistic_low = statistic_observations[index_low]
        bootstrap_statistic_high = statistic_observations[index_high]
        if self == CIMethod.PERCENTILE:
            return bootstrap_statistic_low, bootstrap_statistic_high

        assert self == CIMethod.PIVOTAL
        return 2 * org_statistic - bootstrap_statistic_high, 2 * org_statistic - bootstrap_statistic_low


class FuseMetricConfidenceInterval(FuseMetricBase):
    """
    Wrapper Metric to compute the confidence interval of another metric
    """

    def __init__(self,
                 metric: FuseMetricBase,
                 num_of_bootstraps: int = 10000,
                 rnd_seed: int = 1234,
                 conf_interval: float = 95,
                 stratum_name: Optional[str] = None,
                 ci_method: CIMethod = CIMethod.PERCENTILE) -> None:
        """
        :param metric: metric to compute the confidence interval for
        :param num_of_bootstraps: number of bootstrapping
        :param rng_seed: seed for random number generator.
        :param conf_interval: Confidence interval. Default is 95.
        :param stratum_name: if sampling should be done by strata, specify the key in batch_dict to collect data from
        :param ci_method: specifies the method for computing the confidence intervals from bootstrap samples

        """
        self._metric = metric
        if stratum_name is not None:
            self._metric.add_key_to_collect(name='stratum', key=stratum_name)
        self._num_of_bootstraps = num_of_bootstraps
        self._rnd_seed = rnd_seed
        self._conf_interval = conf_interval
        self._ci_method = ci_method

    def reset(self) -> None:
        """
        Resets collected data for metrics
        :return: None
        """
        self._metric.reset()
        pass

    def collect(self,
                batch_dict: Dict) -> None:
        """
        Calls the metric collect, to gather the data form the batch
        :param batch_dict:
        """
        self._metric.collect(batch_dict)
        pass

    def process(self) -> Dict[str, float]:
        """
        Calculate Confidence Interval for the metric

        :return: dictionary including Area under ROC curve (floating point in range [0, 1]/-1) for each class (class vs rest),
                 -1 will be set for invalid/undefined result (an error message will be printed)
                 The dictionary will also include the average AUC
        """

        def _compute_stats(orig, samples):
            confidence_lower, confidence_upper = self._ci_method.compute_confidence_interval(orig, samples, self._conf_interval)
            return {'org': orig, 'mean': np.mean(samples), 'std': np.std(samples),
                    'conf_interval': self._conf_interval, 'conf_lower': confidence_lower, 'conf_upper': confidence_upper}

        rnd = np.random.RandomState(self._rnd_seed)
        original_sample_results = self._metric.process()
        boot_results = []
        ci_results = {}

        sampled_metric: FuseMetricBase = deepcopy(self._metric)
        sampled_data = sampled_metric.collected_data
        orig_data = {}
        # make sure data is saved in arrays, also keep the original data in array
        for key, data in sampled_data.items():
            orig_data[key] = np.array(data)
            sampled_data[key] = np.empty(orig_data[key].shape)

        stratum_id = orig_data['stratum'] if 'stratum' in orig_data else np.ones(len(orig_data['target']))
        unique_strata = np.unique(stratum_id)

        for bt in range(self._num_of_bootstraps):
            for stratum in unique_strata:
                stratum_filter = stratum_id == stratum
                n_stratum = sum(stratum_filter)
                sample_ix = rnd.randint(0, n_stratum, size=n_stratum)
                for key, data in sampled_data.items():
                    sampled_data[key][stratum_filter] = orig_data[key][stratum_filter][sample_ix]
            boot_results.append(sampled_metric.process())
        # results can be either a list of floats or a list of dictionaries
        if isinstance(original_sample_results, dict):
            for key, orig_val in original_sample_results.items():
                sampled_vals = [sample[key] for sample in boot_results]
                ci_results[key] = _compute_stats(orig_val, sampled_vals)
        elif isinstance(original_sample_results, float):
            ci_results[key] = _compute_stats(original_sample_results, boot_results)

        return ci_results


if __name__ == '__main__':
    from fuse.metrics.classification.metric_auc import FuseMetricAUC
    from fuse.metrics.classification.metric_accuracy import FuseMetricAccuracy

    data = {'preds': np.array([[0.8, 0.1, 0.1],
                               [0.5, 0.3, 0.2],
                               [0.6, 0.3, 0.1],
                               [0.6, 0.1, 0.3],
                               [0.7, 0.2, 0.1],
                               [0.3, 0.2, 0.5],
                               [0.1, 0.2, 0.7],
                               [0.2, 0.6, 0.2],
                               [0.3, 0.3, 0.4],
                               [0.7, 0.2, 0.1],
                               [0.3, 0.2, 0.5],
                               [0.1, 0.2, 0.7],
                               [0.2, 0.6, 0.2],
                               [0.3, 0.3, 0.4],
                               [0.7, 0.2, 0.1]]),
            'targets': np.array([0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 1, 1, 2, 2, 2]),
            'stratum': np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])}

    auc_metric = FuseMetricAUC(pred_name='preds', target_name='targets')
    metric = FuseMetricConfidenceInterval(auc_metric, num_of_bootstraps=10, rnd_seed=198654, stratum_name='stratum')
    metric.collect(data)
    res = metric.process()
    print("AUC with stratum")
    for k, v in res.items():
        print(k, '\t', v)

    auc_metric = FuseMetricAUC(pred_name='preds', target_name='targets')
    metric = FuseMetricConfidenceInterval(auc_metric, num_of_bootstraps=10, rnd_seed=198654)
    metric.collect(data)
    res = metric.process()
    print("AUC no stratum")
    for k, v in res.items():
        print(k, '\t', v)

    data = {'preds': np.array([[0.8, 0.1, 0.1],
                               [0.5, 0.3, 0.2],
                               [0.6, 0.3, 0.1],
                               [0.6, 0.1, 0.3],
                               [0.7, 0.2, 0.1],
                               [0.3, 0.2, 0.5],
                               [0.1, 0.2, 0.7],
                               [0.2, 0.6, 0.2],
                               [0.3, 0.3, 0.4],
                               [0.7, 0.2, 0.1]]),
            'targets': np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2]),
            'weights': np.array([0.03, 0.9, 0.05, 0.52, 0.23, 0.72, 0.13, 0.113, 0.84, 0.09])}

    accuracy_metric = FuseMetricAccuracy(pred_name='preds', target_name='targets', use_sample_weights=True,
                                         sample_weight_name='weights')
    metric = FuseMetricConfidenceInterval(accuracy_metric, 10, 5)
    metric.collect(data)
    res = metric.process()
    print("Accuracy")
    for k, v in res.items():
        print(k, '\t', v)

    data = {'preds': np.array([[0.8, 0.1, 0.1],
                               [0.5, 0.3, 0.2],
                               [0.6, 0.3, 0.1],
                               [0.6, 0.1, 0.3],
                               [0.7, 0.2, 0.1],
                               [0.3, 0.2, 0.5],
                               [0.1, 0.2, 0.7],
                               [0.2, 0.6, 0.2],
                               [0.3, 0.3, 0.4],
                               [0.7, 0.2, 0.1]]),
            'targets': np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2]),
            'weights': np.array([0.03, 0.9, 0.05, 0.52, 0.23, 0.72, 0.13, 0.113, 0.84, 0.09]),
            'stratum': np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 1])}

    accuracy_metric = FuseMetricAccuracy(pred_name='preds', target_name='targets')
    metric = FuseMetricConfidenceInterval(accuracy_metric, 10, 5, stratum_name='stratum')
    metric.collect(data)
    res = metric.process()
    print("Accuracy with stratum, no weights")
    for k, v in res.items():
        print(k, '\t', v)