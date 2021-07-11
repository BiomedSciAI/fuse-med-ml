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

from abc import ABC
import logging
from torch import Tensor
from typing import Union, Dict, Callable, Optional

from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseMetricBase(ABC):
    """
    Base class for Fuse metrics
    """

    def __init__(self,
                 pred_name: str,
                 target_name: str,
                 filter_func: Optional[Callable] = None,
                 use_sample_weights: Optional[bool] = False,
                 **additional_collect: str) -> None:
        """
        Base class for metrics.
        :param pred_name:           batch_dict key for predicted output (e.g., class probabilities after softmax)
        :param target_name:         batch_dict key for target (e.g., ground truth label)
        :param filter_func:         Optional - function that filters batch_dict. The function gets as input batch_dict and returns filtered batch_dict.
                                    When None (default), the entire batch dict is collected and processed.
        :param use_sample_weights:  Optional - when True (default is False), metrics are computed with sample weigths.
                                    For this, use argument 'sample_weight_name' to define the batch_dict key for the sample weight
        :param additional_collect:  args ending with '_name' will be collected into self.collected_data[].
                                    the prefix of argument is the key, and the value of the argument is the key in batch_dict to collect the data from.
                                    e.g., weights_name='data.metadata.weight':
                                            collect data from batch_dict['data.metadata.weight'] into self.collected_data['weights'] array
        """
        additional_collect.update({'pred_name': pred_name, 'target_name': target_name})
        self.key_to_collect: dict = {}
        self.collected_data: Dict[str, list] = {}
        for name_arg, key in additional_collect.items():
            if key is None:  # if user did not pass values for key
                continue
            assert name_arg.endswith('_name')
            name = name_arg[:-5]
            self.add_key_to_collect(name, key)
        if use_sample_weights:
            assert 'sample_weight' in self.key_to_collect, \
                "Metric is expected to use sample weights, but 'sample_weight_name' was not specified"
        self.use_sample_weights = use_sample_weights
        self.filter_func = filter_func
        self.reset()

    def collect(self,
                batch_dict: Dict) -> None:
        """
        Triggered after each batch - collects data for metrics. data are saved into self.collected_data dictionary.
        The dictionary contains at least the keys: predictions, targets.
        Any additional keys to collect from batch_dict can be specified by additional_collect arguments in the init()

        :return: None
        """
        if self.filter_func is not None:
            batch_dict = self.filter_func(batch_dict)

        for name, key in self.key_to_collect.items():
            values = FuseUtilsHierarchicalDict.get(batch_dict, key)
            if isinstance(values, Tensor):
                values = values.detach().cpu().numpy()
            self.collected_data[name].extend(values)

        pass

    def process(self) -> Union[float, Dict[str, float], str]:
        """
        Triggered after each epoch - calculates epoch metrics
        :return:
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Resets collected data for metrics
        :return: None
        """
        for key in self.collected_data.keys():
            self.collected_data.update({key: []})
        pass

    # backward comparability: keep the main collected data in lists
    @property
    def epoch_preds(self):
        return self.collected_data['pred']
    @property
    def epoch_targets(self):
        return self.collected_data['target']

    def add_key_to_collect(self, name, key) -> None:
        """
        Utility function to add keys that should be collected on collect().
        Can be used after the initialization of the class, and before collect() was called (or after reset()).
        :return:  None
        """
        if name in self.key_to_collect:
            if self.key_to_collect[name] != key:
                logging.getLogger('Fuse').debug(f"Replacing key in batch_dict for {name}: was {self.key_to_collect[name]} now {key}")
        self.key_to_collect.update({name: key})
        self.collected_data.update({name: []})
        pass