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

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Hashable, Optional, Sequence, Tuple, Union
import copy
from fuse.utils import uncollate
import torch.distributed as dist
import pandas as pd
import torch
import numpy as np
import scipy

from fuse.utils import NDict


class MetricBase(ABC):
    """
    Required interface for a metric implementation
    """

    @abstractmethod
    def collect(self, batch: Dict) -> None:
        """
        aggregate data from batches
        :param batch: bath dict
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, data: pd.DataFrame) -> None:
        """
        set entire data at once
        :param data: dataframe representation of the data. Each line is a sample and each column is a value field.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        reset state
        """
        raise NotImplementedError

    @abstractmethod
    def eval(
        self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None
    ) -> Union[Dict[str, Any], Any]:
        """
        evaluate the collected data
        :param results: results aggergated by the previous metrics
        :param ids: sequence of sample ids
        The evaluated results should be stored in results. (recommendation: store it under metrics namespace)
        """
        raise NotImplementedError


class MetricCollector(MetricBase):
    """
    Collect data for metrics with native support for data sampling
    """

    DEFAULT_ID_KEYS = ["id", "data.sample_id", "sample_id"]

    def __init__(
        self,
        pre_collect_process_func: Optional[Callable] = None,
        post_collect_process_func: Optional[Callable] = None,
        **keys_to_collect: Dict[str, str],
    ):
        """
        :param pre_collect_process_func: Optional callable - the callable will get as an input a sample_dict and can preprocess it if required
        :param post_collect_process_func: Optional callable - custom process func that convert the fields to be collected to the values that will actually be collected
                                          the callable will get as an input the collected values of a single sample
                                          and can return either a single value or dictionary. The returned values will be collected under the name "post_args"
        :param keys_to_collect: specify the keys you want to collect from the source data
        """
        super().__init__()
        # store input
        self._pre_collect_process_func = pre_collect_process_func
        self._post_collect_process_func = post_collect_process_func
        self._keys_to_collect = copy.copy(keys_to_collect)
        self._id_keys = MetricCollector.DEFAULT_ID_KEYS

        # reset
        self.reset()

    def collect(self, batch: Dict) -> None:
        """
        See super class
        """
        if not isinstance(batch, NDict):
            batch = NDict(batch)

        samples = uncollate(batch)

        # If in distributed mode (multi gpu training) we shall gather the result from all the machine to evaluate with respect to the entire batch.
        if dist.is_initialized():
            world_size = dist.get_world_size()  # num of gpus
            samples_gather = [None for rank in range(world_size)]
            # samples_gather[i] will have the 'samples' value of the i's GPU
            dist.all_gather_object(samples_gather, samples)

            # union all the GPU's samples into one samples list
            samples = []
            for rank in range(world_size):
                samples += samples_gather[rank]

        for sample in samples:
            sample_to_collect = {}

            if self._pre_collect_process_func is not None:
                sample = NDict(self._pre_collect_process_func(sample))

            for name, key in self._keys_to_collect.items():
                value = sample[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()

                sample_to_collect[name] = value

            if self._post_collect_process_func is not None:
                sample_to_collect = {"post_args": self._post_collect_process_func(**sample_to_collect)}

            # store it - assumes batch dimension? What about single sample?
            for name in sample_to_collect:
                self._collected_data[name].append(sample_to_collect[name])

        # extract ids and store it in self._collected_ids
        ids = None
        for key in self._id_keys:
            if key in batch:
                ids = batch[key]
                break

        if ids is not None:
            self._collected_ids.extend(ids)

    @staticmethod
    def _df_dict_apply(data: pd.Series, func: Callable) -> pd.Series:
        result = func(NDict(data.to_dict()))
        return pd.Series(result.flatten())

    @staticmethod
    def _df_dict_apply_kwargs(data: pd.Series, func: Callable) -> pd.Series:
        result = func(**data.to_dict())
        return pd.Series(result.flatten())

    def set(self, data: pd.DataFrame) -> None:
        """
        see super class
        """
        self.reset()

        if self._pre_collect_process_func is not None:
            pre_collect_process = lambda x: self._df_dict_apply(x, self._pre_collect_process_func)
            data = data.apply(pre_collect_process, axis=1)

        data_to_collect = pd.DataFrame(data=None, columns=self._keys_to_collect)
        for name, key in self._keys_to_collect.items():
            if key not in data.keys():
                raise Exception(f"Error key {key} wasn't found. Available keys {data.keys()}")

            data_to_collect[name] = data[key]

        if self._post_collect_process_func is not None:
            post_collect_process = lambda x: self._df_dict_apply_kwargs(x, self._post_collect_process_func)
            data_to_collect = data.apply(post_collect_process, axis=1)

        for name in data_to_collect.keys():
            values = data_to_collect.loc[:, name]
            self._collected_data[name].extend(values)

        # extract ids and store it in self._collected_ids
        ids = None
        for key in self._id_keys:
            if key in data.keys():
                ids = list(data[key])
                break

        if ids is not None:
            self._collected_ids.extend(ids)

    def reset(self) -> None:
        """
        See super class
        """
        if self._post_collect_process_func is None:
            self._collected_data = {name: [] for name in self._keys_to_collect}
        else:
            # collect everything you get from post_collect_process_args
            self._collected_data = {"post_args": []}

        self._collected_ids = []  # the original collected ids

        self._sampled_ids = None  # the required ids - set be sample() method

    def get_ids(self) -> Sequence[Hashable]:
        """
        See super class
        """
        return self._collected_ids

    def get(self, ids: Optional[Sequence[Hashable]] = None) -> Tuple[Dict[str, Any]]:
        """
        Get collected data - collected data dictionary and collected ids.
        each element in the dictionary will be a list of values from all samples
        """
        if ids is None:
            return copy.copy(self._collected_data)
        else:
            # convert required ids to permutation
            original_ids = self._collected_ids
            required_ids = ids
            permutation = [original_ids.index(sample_id) for sample_id in required_ids]

            # create the permuted dictionary
            data = {}
            for name, values in self._collected_data.items():
                data[name] = [values[i] for i in permutation]

            return data

    def eval(self, results: Dict[str, Any] = None) -> Union[Dict[str, Any], Any]:
        """
        Empty implementation - do nothing
        """
        pass


class MetricWithCollectorBase(MetricBase):
    """
    Base implementation of metric with built-in collector
    There is also an option for an external data collector  - in a case space optimization required and there by using shared collector for few metrics
    """

    def __init__(
        self,
        pre_collect_process_func: Optional[Callable] = None,
        post_collect_process_func: Optional[Callable] = None,
        external_data_collector: Optional[MetricCollector] = None,
        extract_ids: bool = False,
        **kwargs,
    ) -> None:
        """
        :param pre_collect_process_func: Optional callable - the callable will get as an input a batch_dict or a dataframe and can preprocess it if required
        :param external_data_collector: Optional - in a case space optimization required and there by using shared collector for few metrics
        :param extract_ids: self._extract_arguments packs all arguments for a underlying function. Set to True, to pack also the ids (under the name 'ids')
        :param kwargs: specify keywords and value arguments you want to collect from the source data.
                can be strings (key names) and/or actual values
                to collect from the results dictionary: add a "results:" prefix to the key name
        """
        super().__init__()
        self._keys_to_collect = {n: k for n, k in kwargs.items() if isinstance(k, str) and not k.startswith("results:")}
        self._keys_from_results = {
            n: k[len("results:") :] for n, k in kwargs.items() if isinstance(k, str) and k.startswith("results:")
        }
        self._value_args = {n: k for n, k in kwargs.items() if k is not None and not isinstance(k, str)}
        self._collector = (
            MetricCollector(pre_collect_process_func, post_collect_process_func, **self._keys_to_collect)
            if external_data_collector is None
            else external_data_collector
        )
        self._collect_data_flag = external_data_collector is None
        self._extract_ids = extract_ids

    def collect(self, batch: Dict) -> None:
        """
        See super class
        """
        if self._collect_data_flag:
            return self._collector.collect(batch)

    def set(self, data: pd.DataFrame) -> None:
        """
        See super class
        """
        if self._collect_data_flag:
            return self._collector.set(data)

    def reset(self) -> None:
        """
        See super class
        """
        if self._collect_data_flag:
            return self._collector.reset()

    def _extract_arguments(self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None) -> Dict:
        """
        extract keyworded arguments and value arguments from collected data and results dict
        """
        if not isinstance(results, NDict):
            results = NDict(results)

        arg_dict = {}
        data = self._collector.get(ids)
        arg_dict.update(data)

        for name in self._keys_from_results:
            res = results[self._keys_from_results[name]]
            if isinstance(res, Callable):  # per-sample metric
                res = res(ids)
            arg_dict[name] = res

        for name in self._value_args:
            arg_dict[name] = self._value_args[name]

        # pack also ids if those are required - see details in self.__init__
        if self._extract_ids:
            if ids is None:
                arg_dict["ids"] = self._collector.get_ids()
            else:
                arg_dict["ids"] = ids

        return arg_dict

    @abstractmethod
    def eval(self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None) -> None:
        """
        See super class
        """
        raise NotImplementedError


class MetricDefault(MetricWithCollectorBase):
    """
    Default generic implementation for metric
    Can be used for any metric getting as an input list of prediction, list of targets and optionally additional parameters
    """

    def __init__(self, metric_func: Callable, pred: Optional[str] = None, target: Optional[str] = None, **kwargs):
        """
        :param pred: prediction key to collect
        :param target: target key to collect
        :param metric_func: function getting as a input list of predictions, targets and optionally more arguments specified in kwargs
                            the function should return a single result or a dictionary of results
        :param kwargs: additional keyword arguments for MetricWithCollectorBase.
                       The keyword expected to be an argument name of metric_func and the value a string that is a key to value store in batch dict.
                       If instead a value should be extracted from results dict use "results:<key in results dict>
                       Example:
                       def my_metric_func(pred, target, sample_weight, operation_point):
                           ...
                       metric = MetricDefault(metric_func=my_metric_func,
                                              pred="model.classification.preds", target="data.classification.labels",
                                              sample_weight="data.classification.weight", operation_point="results:metrics.operation_point")
        """
        super().__init__(pred=pred, target=target, **kwargs)
        self._metric_func = metric_func

    def eval(self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None) -> Union[Dict, Any]:
        """
        See super class
        """

        # extract values from collected data and results dict
        kwargs = self._extract_arguments(results, ids)

        # single evaluation method
        return self._metric_func(**kwargs)


class MetricPerSampleDefault(MetricWithCollectorBase):
    """
    Default generic implementation for a case that is better to compute the metric per sample and then aggregate the results.
    Can be used for any metric getting as an input list of prediction, list of targets and optionally additional parameters
    """

    def __init__(
        self, pred: str, target: str, metric_per_sample_func: Callable, result_aggregate_func: Callable, **kwargs
    ):
        """
        :param pred: prediction key to collect
        :param target: target key to collect
        :param metric_per_sample_func: function that gets as input the values to collect specified in metric constructor, typically pred and target.
                                       A sequence of all the returned values from all samples will be passed to result_aggregate_func
        :param result_aggregate_func: function that get the output of metric_per_sample_func and aggregates it over multiple samples
        :param kwargs: additional kw arguments for MetricWithCollectorBase
        """

        super().__init__(pred=pred, target=target, post_collect_process_func=metric_per_sample_func, **kwargs)
        self._result_aggregate_func = result_aggregate_func

    def eval(self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None) -> Union[Dict, Any]:
        """
        See super class
        """

        # extract values from collected data and results dict
        kwargs = self._extract_arguments(results, ids)

        # single evaluation method
        return self._result_aggregate_func(kwargs["post_args"])


class GroupAnalysis(MetricWithCollectorBase):
    """
    Evaluate a metric per group and compute basic statistics about the different per group results.
    eval() method returns a dictionary of the following format:
    {'mean': <>, 'std': <>, 'median': <>, <group 0>: <>, <group 1>: <>, ...}
    """

    def __init__(self, metric: MetricBase, group: str, **super_kwargs) -> None:
        """
        :param metric: metric to analyze
        :param group: key to extract the group from
        :param super_kwargs: additional arguments for super class (MetricWithCollectorBase) constructor
        """
        super().__init__(group=group, **super_kwargs)
        self._metric = metric

    def collect(self, batch: Dict) -> None:
        "See super class"
        self._metric.collect(batch)
        return super().collect(batch)

    def set(self, data: pd.DataFrame) -> None:
        "See super class"
        self._metric.set(data)
        return super().set(data)

    def reset(self) -> None:
        "See super class"
        self._metric.reset()
        return super().reset()

    def eval(self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None) -> Dict[str, Any]:
        """
        See super class
        :return: a dictionary of the following format {'mean': <>, 'std': <>, 'median': <>, <group 0>: <>, <group 1>: <>, ...}
        """
        data = self._collector.get(ids)
        if ids is None:
            ids = self._collector.get_ids()
        if not ids:
            raise Exception(
                "Error: group analysis is supported only when a unique identifier is specified. Add key 'id' to your data"
            )
        ids = np.array(ids)

        groups = np.array(data["group"])
        unique_groups = set(groups)

        group_analysis_results = {}
        for group_value in unique_groups:
            group_ids = ids[groups == group_value]

            group_analysis_results[str(group_value)] = self._metric.eval(results, group_ids)

        # compute stats
        group_results_list = list(group_analysis_results.values())
        if isinstance(group_results_list[0], dict):  # multiple values
            # get all keys
            all_keys = set()
            for group_result in group_results_list:
                all_keys |= set(group_result.keys())

            for key in all_keys:
                values = [group_result[key] for group_result in group_results_list]
                try:
                    group_analysis_results[f"{key}.mean"] = np.mean(values)
                    group_analysis_results[f"{key}.std"] = np.std(values)
                    group_analysis_results[f"{key}.median"] = np.median(values)
                except:
                    # do nothing
                    pass
        else:  # single value
            values = [group_result for group_result in group_results_list]
            try:
                group_analysis_results["mean"] = np.mean(values)
                group_analysis_results["std"] = np.std(values)
                group_analysis_results["median"] = np.median(values)
            except:
                # do nothing
                pass

        return group_analysis_results


class Filter(MetricWithCollectorBase):
    """
    Evaluate a sub-group of data. This utility will filter non relevant samples and will call to the given metric.
    """

    def __init__(self, metric: MetricBase, filter: str, **super_kwargs) -> None:
        """
        :param metric: metric to filter samples for
        :param group: key to extract filter
        :param super_kwargs: additional arguments for super class (MetricWithCollectorBase) constructor
        """
        super().__init__(filter=filter, **super_kwargs)
        self._metric = metric

    def collect(self, batch: Dict) -> None:
        "See super class"
        self._metric.collect(batch)
        return super().collect(batch)

    def set(self, data: pd.DataFrame) -> None:
        "See super class"
        self._metric.set(data)
        return super().set(data)

    def reset(self) -> None:
        "See super class"
        self._metric.reset()
        return super().reset()

    def eval(self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None) -> Dict[str, Any]:
        """
        See super class
        :return: a dictionary of the following format {'mean': <>, 'std': <>, 'median': <>, <group 0>: <>, <group 1>: <>, ...}
        """
        data = self._collector.get()
        if ids is None:
            ids = self._collector.get_ids()
        if not ids:
            raise Exception(
                "Error: group analysis is supported only when a unique identifier is specified. Add key 'id' to your data"
            )
        ids = np.array(ids)

        filter = np.array(data["filter"])
        keep_ids = ids[~filter]
        metric_result = self._metric.eval(results, keep_ids)
        return metric_result


class CI(MetricWithCollectorBase):
    """
    Compute confidence interval for a metric
    eval() method returns a dictionary of the following format:
    {'org': <>, 'mean': <>, 'std': <>, 'conf_interval': <>, 'conf_lower': <>, 'conf_upper': <>}
    """

    def __init__(
        self,
        metric: MetricBase,
        stratum: str,
        num_of_bootstraps: int = 10000,
        rnd_seed: int = 1234,
        conf_interval: float = 95,
        ci_method: str = "PERCENTILE",
        **super_kwargs,
    ) -> None:
        """
        :param metric: metric to compute the confidence interval for
        :param stratum: if sampling should be done by strata, specify the key in batch_dict to collect data from
        :param num_of_bootstraps: number of bootstrap iterations
        :param rnd_seed: seed for random number generator.
        :param conf_interval: Confidence interval. Default is 95.
        :param ci_method: specifies the method for computing the confidence intervals from bootstrap samples. Options: NORMAL (assuming normal distribution), PERCENTILE, PIVOTAL
        """
        super().__init__(stratum=stratum, **super_kwargs)

        self._metric = metric
        self._num_of_bootstraps = num_of_bootstraps
        self._rnd_seed = rnd_seed
        self._conf_interval = conf_interval
        self._ci_method = ci_method

        # verify
        assert ci_method in [
            "NORMAL",
            "PERCENTILE",
            "PIVOTAL",
        ], f"Error: unexpected confidence interval method: {ci_method}"

    def collect(self, batch: Dict) -> None:
        "See super class"
        self._metric.collect(batch)
        return super().collect(batch)

    def set(self, data: pd.DataFrame) -> None:
        "See super class"
        self._metric.set(data)
        return super().set(data)

    def reset(self) -> None:
        "See super class"
        self._metric.reset()
        return super().reset()

    def eval(self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None) -> Dict[str, Any]:
        """
        See super class
        :return: dictionary of format - {'org': <>, 'mean': <>, 'std': <>, 'conf_interval': <>, 'conf_lower': <>, 'conf_upper': <>}
        """
        data = self._collector.get()
        if ids is None:
            ids = self._collector.get_ids()
        if not ids:
            raise Exception(
                "Error: confidence interval is supported only when a unique identifier is specified. Add key 'id' to your data"
            )
        ids = np.array(ids)

        rnd = np.random.RandomState(self._rnd_seed)
        original_sample_results = self._metric.eval(results)
        boot_results = []
        ci_results = {}

        stratum_id = np.array(data["stratum"]) if "stratum" in data else np.ones(len(data["index"]))
        unique_strata = np.unique(stratum_id)

        for _ in range(self._num_of_bootstraps):
            sampled_ids = ids.copy()
            for stratum in unique_strata:
                stratum_filter = stratum_id == stratum
                n_stratum = sum(stratum_filter)
                random_sample = rnd.randint(0, n_stratum, size=n_stratum)
                sampled_ids[stratum_filter] = ids[stratum_filter][random_sample]
            boot_results.append(self._metric.eval(results, sampled_ids))

        # results can be either a list of floats or a list of dictionaries
        if isinstance(original_sample_results, dict):
            for key, orig_val in original_sample_results.items():
                sampled_vals = [sample[key] for sample in boot_results]
                try:
                    ci_results[key] = self._compute_stats(self._ci_method, orig_val, sampled_vals, self._conf_interval)
                except:
                    ci_results[key] = orig_val
        elif isinstance(original_sample_results, float):
            ci_results = self._compute_stats(
                self._ci_method, original_sample_results, boot_results, self._conf_interval
            )

        return ci_results

    @staticmethod
    def _compute_stats(
        ci_method: str, orig: Union[float, np.ndarray], samples: Sequence[Union[float, np.ndarray]], confidence: float
    ) -> Dict[str, Union[np.ndarray, float]]:
        "Compute and package into dictionary CI results"
        confidence_lower, confidence_upper = CI.compute_confidence_interval(ci_method, orig, samples, confidence)
        return {
            "org": orig,
            "mean": np.mean(samples),
            "std": np.std(samples),
            "conf_interval": confidence,
            "conf_lower": confidence_lower,
            "conf_upper": confidence_upper,
        }

    @staticmethod
    def compute_confidence_interval(
        method: str,
        org_statistic: Union[float, np.ndarray],
        bootstrap_statistics: Sequence[Union[float, np.ndarray]],
        confidence: float,
    ) -> Tuple[float, float]:
        lower_confidence_ratio = (1 - confidence / 100.0) / 2.0
        if method == "NORMAL":
            sigma = np.std(bootstrap_statistics)
            c_alpha = scipy.stats.norm.isf(lower_confidence_ratio)
            return org_statistic - c_alpha * sigma, org_statistic + c_alpha * sigma

        statistic_observations = np.sort(bootstrap_statistics)

        n = len(statistic_observations)
        index_low = int(np.floor(lower_confidence_ratio * n))
        index_high = int(np.floor((1 - lower_confidence_ratio) * n))

        bootstrap_statistic_low = statistic_observations[index_low]
        bootstrap_statistic_high = statistic_observations[index_high]
        if method == "PERCENTILE":
            return bootstrap_statistic_low, bootstrap_statistic_high

        assert method == "PIVOTAL"
        return 2 * org_statistic - bootstrap_statistic_high, 2 * org_statistic - bootstrap_statistic_low
