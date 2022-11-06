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
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F

from fuse.utils import NDict
from fuse.utils.data.collate import CollateToBatchList
from fuse.data import get_sample_id_key


class CollateDefault(CollateToBatchList):
    """
    Default collate_fn method to be used when creating a DataLoader.
    Will collate each value with PyTorch default collate.
    Special collates per key can be specified in special_handlers_keys
    sample_id key will be collected to a list.
    Few options to special handlers implemented in this class as static methods
    """

    def __init__(
        self,
        skip_keys: Sequence[str] = tuple(),
        keep_keys: Sequence[str] = tuple(),
        raise_error_key_missing: bool = True,
        special_handlers_keys: Dict[str, Callable] = None,
    ):
        """
        :param skip_keys: do not collect the listed keys
        :param keep_keys: specifies a list of keys to collect. missing keep_keys are skipped.
        :param special_handlers_keys: per key specify a callable which gets as an input list of values and convert it to a batch.
                                      The rest of the keys will be converted to batch using PyTorch default collate_fn()
                                      Example of such Callable can be seen in the CollateDefault.pad_all_tensors_to_same_size.
        :param raise_error_key_missing: if False, will not raise an error if there are keys that do not exist in some of the samples. Instead will set those values to None.
        """
        super().__init__(skip_keys, raise_error_key_missing)
        self._special_handlers_keys = {}
        if special_handlers_keys is not None:
            self._special_handlers_keys.update(special_handlers_keys)
        self._special_handlers_keys[get_sample_id_key()] = CollateDefault.just_collect_to_list
        self._keep_keys = keep_keys

    def __call__(self, samples: List[Dict]) -> Dict:
        """
        collate list of samples into batch_dict
        :param samples: list of samples
        :return: batch_dict
        """
        batch_dict = NDict()

        # collect all keys
        keys = self._collect_all_keys(samples)
        if self._keep_keys:
            keys = [k for k in keys if k in self._keep_keys]
        # collect values
        for key in keys:

            # skip keys
            if key in self._skip_keys:
                continue

            try:
                # collect values into a list
                collected_values, has_error = self._collect_values_to_list(samples, key)

                # batch values
                self._batch_dispatch(batch_dict, samples, key, has_error, collected_values)
            except:
                print(f"Error: Failed to collect key {key}")
                raise

        return batch_dict

    def _batch_dispatch(
        self, batch_dict: dict, samples: List[dict], key: str, has_error: bool, collected_values: list
    ) -> None:
        """
        dispatch a key into collate function and save it into batch_dict
        :param batch_dict: batch dictionary to update
        :param samples: list of samples
        :param key: key to collate
        :param has_error: True, if the key is missing in one of the samples
        :param collected values: the values collected from samples
        :return: nothing - the new batch will be added to batch_dict
        """
        if has_error:
            # do nothing when error occurs
            batch_dict[key] = collected_values
        elif key in self._special_handlers_keys:
            # use special handler if specified
            batch_dict[key] = self._special_handlers_keys[key](collected_values)
        elif isinstance(collected_values[0], (torch.Tensor, np.ndarray, float, int, str, bytes)):
            # batch with default PyTorch implementation
            batch_dict[key] = default_collate(collected_values)
        else:
            batch_dict[key] = collected_values

    @staticmethod
    def just_collect_to_list(values: List[Any]):
        """
        special handler doing nothing - will just keep the collected list
        """
        return values

    @staticmethod
    def pad_all_tensors_to_same_size(values: List[torch.Tensor], pad_val: float = 0.0) -> torch.Tensor:
        """
        pad tensors and create a batch - the shape will be the max size per dim
        values: list of tensor - all should have the same number of dimensions
        pad_val: constant value for padding
        :return: torch.stack of padded tensors
        """

        # verify all are tensor and that they have the same dim size
        assert isinstance(values[0], torch.Tensor), f"Expecting just tensors, got {type(values[0])}"
        num_dims = len(values[0].shape)
        for value in values:
            assert isinstance(value, torch.Tensor), f"Expecting just tensors, got {type(value)}"
            assert (
                len(value.shape) == num_dims
            ), f"Expecting all tensors to have the same dim size, got {len(value.shape)} and {num_dims}"

        # get max per dim
        max_per_dim = np.amax(np.stack([value.shape for value in values]), axis=0)

        # pad
        def _pad_size(value, dim):
            assert max_per_dim[dim] >= value.shape[dim]
            return [0, max_per_dim[dim] - value.shape[dim]]

        padded_values = []

        for value in values:
            padding = []
            # F.pad padding description is expected to be provided in REVERSE order (see torch.nn.functional.pad doc)
            for dim in reversed(range(num_dims)):
                padding += _pad_size(value, dim)
            padded_value = F.pad(value, padding, mode="constant", value=pad_val)
            padded_values.append(padded_value)

        return default_collate(padded_values)
