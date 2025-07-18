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

import math
from collections.abc import Sequence
from typing import Any, Dict, List

import numpy as np
from torch.utils.data.sampler import BatchSampler, Sampler

from fuse.data.datasets.dataset_base import DatasetBase


class BatchSamplerDefault(BatchSampler):
    """
    Torch batch sampler - balancing per batch
    """

    def __init__(
        self,
        dataset: DatasetBase,
        balanced_class_name: str | None,
        num_balanced_classes: int = None,
        sampler: Sampler | None = None,
        batch_size: int | None = None,
        mode: str = "exact",
        balanced_class_weights: List[int]
        | List[float]
        | Dict[Any, int]
        | Dict[Any, float]
        | None = None,
        num_batches: int | None = None,
        verbose: bool = False,
        **dataset_get_multi_kwargs: dict,
    ) -> None:
        """
        :param dataset: dataset used to extract the balanced class from each sample
        :param balanced_class_name:  the name of balanced class to extract from dataset.
                                     Set to none if you just want to randomly sample from entire dataset.
        :param num_balanced_classes: Not used. the number of classes inferred automatically. Keeping for backward compatibility.
        :param sampler: Optional - pytorch sampler for collecting the data.
                        In use in DDP strategy, when PL trainer re-instantiate a custom batch_sampler with a DistributedSampler.
        :param batch_size: batch_size.
                        - In "exact" mode
                            If balanced_class_weights is None, must be set and divided by number of classes. Otherwise keep None.
                        - In "approx" mode
                          Must be set
        :param mode: either 'exact' or 'approx'. if 'exact each element in balanced_class_weights will specify the exact number of samples from this class.
                        if 'approx' - each element will specify the a probability that a sample will be from this class
        :param balanced_class_weights: Optional, integer/float per balanced class.
                                        If it's a list, the expected length is number of classes and the classes expected to be sequential integers.
                                        If it's a dictionary: expected to include <key = balanced class value: value: weight> and classes could be any value - for example  strings.
                                        In mode 'exact' the weights should be integers that sums up to batch size.
                                        In mode 'approx' the weights should be floats that sums up to ~1
                                        If not specified, equal number of samples from each class will be used.
        :param num_batches: optional - if set to an int will force num_batches,
                                       if set to None or "over_sample"  - num_batches will be automatically to go over each sample at least once (exactly or approximately).
                                       if set to "down_sample"  - num_batches will be automatically to go over each sample in one of the classes (exactly or approximately) and the rest will be down sampled.
        :param verbose:
        :param dataset_get_multi_kwargs: extra parameters for dataset.get_multi() to optimize the running time.
        """
        super().__init__(sampler, batch_size, False)

        # store input
        self._dataset = dataset
        self._balanced_class_name = balanced_class_name
        self._sampler = sampler
        self._batch_size = batch_size
        self._mode = mode
        if isinstance(balanced_class_weights, Sequence):
            self._balanced_class_weights = dict(enumerate(balanced_class_weights))
        else:
            self._balanced_class_weights = balanced_class_weights

        self._num_batches = num_batches
        self._dataset_get_multi_kwargs = dataset_get_multi_kwargs.copy()
        self._verbose = verbose
        # modify relevant keys
        if self._balanced_class_name is not None:
            if "keys" not in self._dataset_get_multi_kwargs:
                self._dataset_get_multi_kwargs["keys"] = [self._balanced_class_name]

        # read data from dataset
        if self._sampler:
            if self._verbose:
                print(
                    f"BatchSamplerDefault got a sampler of type: {type(self._sampler).__name__}"
                )
            # Get all samples that sampler posses.
            # In use in DDP strategy: each process runs on a different GPU with a different instance of 'DistributedSampler'.
            #   The DistributedSampler(s) make sure that each GPU posses a different subset of the samplers to avoid overlaps.
            items = list(self._sampler)  # get all samples that sampler posses
        else:
            items = None  # equivalent to all samples in dataset

        # take a subset of the dataset with the relevant values
        dataset.subset(items)

        # get balanced classes per each sample
        if self._balanced_class_name is not None:
            collected_data = dataset.get_multi(
                None, desc="batch_sampler", **self._dataset_get_multi_kwargs
            )
            self._balanced_classes = self._extract_balanced_classes(collected_data)
        else:
            self._balanced_classes = np.zeros(len(self._dataset), dtype=np.int32)

        self._balanced_class_values = np.unique(self._balanced_classes)
        if self._balanced_class_weights is None:
            self._balanced_class_weights_list = None
        else:
            # add missing weights (set probability / num instances in a batch to zero)
            self._balanced_class_weights = {
                value: self._balanced_class_weights.get(value, 0)
                for value in self._balanced_class_values
            }
            self._balanced_class_weights_list = [
                self._balanced_class_weights.get(value, 0)
                for value in self._balanced_class_values
            ]

        # validate input
        # modes
        if self._mode not in ["exact", "approx"]:
            raise Exception(
                "Error, expected sampler mode to be either 'exact' or 'approx', got {mode}"
            )

        # weights
        if self._mode == "exact":
            if self._balanced_class_weights_list is not None:
                for weight in self._balanced_class_weights_list:
                    if not isinstance(weight, int):
                        raise Exception(
                            f"Error: in mode 'exact', expecting only integers in balanced_class_weights, got {type(weight)}"
                        )
                if self._batch_size is not None:
                    if self._batch_size != sum(self._balanced_class_weights_list):
                        raise Exception(
                            f"Error: in mode 'exact', expecting balanced_class_weights {self._balanced_class_weights} to sum up to batch size {self._batch_size}. Consider setting batch_size to None to automatically compute the batch size."
                        )
                else:
                    self._batch_size = sum(self._balanced_class_weights_list)

            elif self._batch_size is None:
                raise Exception(
                    "Error: In 'approx' mode, either batch_size or balanced_class_weights"
                )

        if self._mode == "approx":
            if self._batch_size is None:
                raise Exception("Error: in mode 'approx', batch size must be set.")
            if self._balanced_class_weights_list is not None:
                for weight in self._balanced_class_weights_list:
                    if weight > 1.0:
                        raise Exception(
                            f"Error: in mode 'exact', expecting probabilities, got {weight}"
                        )
                if not math.isclose(sum(self._balanced_class_weights_list), 1.0):
                    raise Exception(
                        f"Error: in mode 'exact', expecting balanced_class_weight to sum up to almost one, got {balanced_class_weights}"
                    )

        # if weights not specified, set weights to equally balance per batch
        if self._balanced_class_weights_list is None:
            if self._mode == "exact":
                self._balanced_class_weights = {
                    cls_i: self._batch_size // len(self._balanced_class_values)
                    for cls_i in self._balanced_class_values
                }
            elif self._mode == "approx":
                self._balanced_class_weights = {
                    cls_i: 1.0 / len(self._balanced_class_values)
                    for cls_i in self._balanced_class_values
                }
                self._balanced_class_weights_list = [
                    self._balanced_class_weights[cls_i]
                    for cls_i in self._balanced_class_values
                ]
        else:
            if self._batch_size < sum(self._balanced_class_weights_list):
                raise Exception(
                    f"Error: batch_size ({self._batch_size}) should be greater than or equal to number of samples specified in balanced_class_weights."
                )

        # split samples to groups
        self._balanced_class_indices = {
            cls_i: np.where(self._balanced_classes == cls_i)[0]
            for cls_i in self._balanced_class_values
        }
        self._balanced_class_sizes = {
            cls_i: len(self._balanced_class_indices[cls_i])
            for cls_i in self._balanced_class_values
        }

        # make sure that size != 0 for all balanced classes
        for cls_ind, cls_size in self._balanced_class_sizes.items():
            if self._balanced_class_weights[cls_ind] != 0.0 and cls_size == 0:
                msg = f"Every balanced class must include at least one sample (num of samples per balanced class{self._balanced_class_sizes} and weights are {self._balanced_class_weights})"
                raise Exception(msg)

        # calc batch index to balanced class mapping according to weights
        if self._mode == "exact":
            self._batch_index_to_class = []
            for balanced_cls in self._balanced_class_weights:
                self._batch_index_to_class.extend(
                    [balanced_cls] * self._balanced_class_weights[balanced_cls]
                )
        else:
            # probabilistic method - will be randomly select per epoch
            self._batch_index_to_class = None

        # Shuffle balanced class indices
        for indices in self._balanced_class_indices.values():
            np.random.shuffle(indices)

        # Calculate num batches. Number of batches to iterate over all data at least once (exactly or approximately)
        # Calculate only if not directly specified by the user
        if self._num_batches is None or self._num_batches in [
            "over_sample",
            "down_sample",
        ]:
            if self._mode == "exact":
                samples_per_batch = self._balanced_class_weights
            else:  # mode is approx
                # approximate size!
                samples_per_batch = {
                    cls_i: val * self._batch_size
                    for cls_i, val in self._balanced_class_weights.items()
                }
            balanced_class_weighted_sizes = [
                math.ceil(self._balanced_class_sizes[cls_i] / samples_per_batch[cls_i])
                if self._balanced_class_weights[cls_i] != 0
                else 0
                for cls_i in self._balanced_class_sizes
            ]
            if self._num_batches == "down_sample":
                balanced_class_weighted_size = min(balanced_class_weighted_sizes)
            else:  # over sample
                balanced_class_weighted_size = max(balanced_class_weighted_sizes)

            self._num_batches = int(balanced_class_weighted_size)

        # pointers per class
        self._cls_pointers = dict.fromkeys(self._balanced_class_values, 0)
        self._sample_pointer = 0

    def __iter__(self) -> np.ndarray:
        for batch_idx in range(self._num_batches):
            yield self._make_batch()

    def __len__(self) -> int:
        return self._num_batches

    def _get_sample(self, balanced_class: Any) -> int:
        """
        Sample index given balanced class value
        :param balanced_class: integer representing balanced class value
        :return: sample index
        """
        sample_idx = self._balanced_class_indices[balanced_class][
            self._cls_pointers[balanced_class]
        ]

        self._cls_pointers[balanced_class] += 1
        if (
            self._cls_pointers[balanced_class]
            == self._balanced_class_sizes[balanced_class]
        ):
            self._cls_pointers[balanced_class] = 0
            np.random.shuffle(self._balanced_class_indices[balanced_class])

        return int(sample_idx)

    def _make_batch(self) -> list:
        """
        :return: list of indices to collate batch
        """
        if self._mode == "exact":
            batch_index_to_class = self._batch_index_to_class
        else:  # mode == approx
            # calc one according to probabilities
            batch_index_to_class = np.random.choice(
                self._balanced_class_values,
                self._batch_size,
                p=self._balanced_class_weights_list,
            )
        batch_sample_indices = []
        for batch_index in range(self._batch_size):
            balanced_class = batch_index_to_class[batch_index]
            batch_sample_indices.append(self._get_sample(balanced_class))

        np.random.shuffle(batch_sample_indices)
        return batch_sample_indices

    def _extract_balanced_classes(self, collected_data: List[dict]) -> np.ndarray:
        """
        Extracting balanced class values from collected data.
        If a special extra logic is required. Either override this method or the logic in Op and append to dataset pipeline
        """
        assert len(collected_data) > 0, "Error: sampling failed, dataset size is 0"
        balanced_classes = [
            sample[self._balanced_class_name] for sample in collected_data
        ]
        return np.array(balanced_classes)
