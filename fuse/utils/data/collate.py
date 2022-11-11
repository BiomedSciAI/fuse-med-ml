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
import logging
from typing import Callable, Dict, List, Sequence, Tuple

from fuse.utils import NDict

import torch
import numpy as np


class CollateToBatchList(Callable):
    """
    Collate list of dictionaries to a batch dictionary, each value in dict will be a list of values collected from all samples.
    """

    def __init__(
        self,
        skip_keys: Sequence[str] = tuple(),
        raise_error_key_missing: bool = True,
    ):
        """
        :param skip_keys: do not collect the listed keys
        :param raise_error_key_missing: if False, will not raise an error if there are keys that do not exist in some of the samples. Instead will set those values to None.
        """
        self._skip_keys = skip_keys
        self._raise_error_key_missing = raise_error_key_missing

    def __call__(self, samples: List[Dict]) -> Dict:
        """
        collate list of samples into batch_dict
        :param samples: list of samples
        :return: batch_dict
        """
        batch_dict = NDict()

        # collect all keys
        keys = self._collect_all_keys(samples)

        # collect values
        for key in keys:

            # skip keys
            if key in self._skip_keys:
                continue

            try:
                # collect values into a list
                collected_values, has_error = self._collect_values_to_list(samples, key)
                batch_dict[key] = collected_values
            except:
                print(f"Error: Failed to collect key {key}")
                raise

        return batch_dict

    def _collect_all_keys(self, samples: List[Dict]):
        """
        collect list of keys used in any one of the samples
        :param samples: list of samples
        :return: list of keys
        """
        keys = set()
        for sample in samples:
            if not isinstance(sample, NDict):
                sample = NDict(sample)
            keys |= set(sample.keypaths())
        return keys

    def _collect_values_to_list(self, samples: List[str], key: str) -> Tuple[List, bool]:
        """
        collect values of given key into a list
        :param samples: list of samples
        :param key: key to collect
        :return: list of values
        """
        has_error = False
        collected_values = []
        for index, sample in enumerate(samples):
            sample = NDict(sample)
            if key not in sample:
                has_error = True
                if self._raise_error_key_missing:
                    raise Exception(f"Error: key {key} does not exist in sample {index}: {sample}")
                else:
                    value = None
            else:
                value = sample[key]
            collected_values.append(value)
        return collected_values, has_error


def uncollate(batch: Dict) -> List[Dict]:
    """
    Reverse collate method
    Gets a batch_dict and convert it back to list of samples
    """
    samples = []
    if not isinstance(batch, NDict):
        batch = NDict(batch)
    keys = batch.keypaths()

    # empty batch
    if not keys:
        return samples

    if "data.sample_id" in keys:
        batch_size = len(batch["data.sample_id"])
    else:
        batch_size = None

    if batch_size is None:
        for key in keys:
            if isinstance(batch[key], torch.Tensor):
                batch_size = len(batch[key])
                break

    if batch_size is None:
        for key in keys:
            if isinstance(batch[key], (np.ndarray, list)):
                batch_size = len(batch[key])
                break

    if batch_size is None:
        return batch  # assuming batch dict with no samples

    for sample_index in range(batch_size):
        sample = NDict()
        for key in keys:
            if isinstance(batch[key], (np.ndarray, torch.Tensor, list)):
                try:
                    sample[key] = batch[key][sample_index]
                except IndexError:
                    logging.error(
                        f"Error - IndexError - key={key}, batch_size={batch_size}, type={type((batch[key]))}, len={len(batch[key])}"
                    )
                    raise
            else:
                sample[key] = batch[key]  # broadcast single value for all batch

        samples.append(sample)

    return samples
