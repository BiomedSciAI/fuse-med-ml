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

"""
Torch batch sampler - balancing per batch
"""
import logging
import math
from typing import Any, List, Optional

import numpy as np
from torch.utils.data.sampler import Sampler

from fuse.data.dataset.dataset_base import FuseDatasetBase
from fuse.utils.utils_debug import FuseUtilsDebug
from fuse.utils.utils_logger import log_object_input_state


class FuseSamplerBalancedBatch(Sampler):
    """
    Torch batch sampler - balancing per batch
    """

    def __init__(self, dataset: FuseDatasetBase, balanced_class_name: str, num_balanced_classes: int, batch_size: int,
                 balanced_class_weights: Optional[List[int]] = None, balanced_class_probs: Optional[List[float]] = None,
                 num_batches: Optional[int] = None, use_dataset_cache: bool = False) -> None:
        """
        :param dataset: dataset used to extract the balanced class from each sample
        :param balanced_class_name:  the name of balanced class to extract from dataset
        :param num_balanced_classes: number of classes to balance between
        :param batch_size: batch_size.
                        - If balanced_class_weights=Nobe, Must be divided by num_balanced_classes
                        - Otherwise must be equal to sum of balanced_class_weights
        :param balanced_class_weights: Optional, integer per balanced class,
                                        specifying the number of samples from each class to include in each batch.
                                        If not specified and equal number of samples from each class will be used.
        :param balanced_class_probs: Optional, probability per class. Random sampling approach will be performed.
                                     such that an epoch will go over all the data at least once.
        :param num_batches: Optional, Set number of batches. If not set. The number of batches will automatically set.

        :param use_dataset_cache: to retrieve the balanced class from dataset try to use caching.
                                 Should be set to True if reading it from cache is faster than running the single processor
        """
        # log object
        log_object_input_state(self, locals())

        super().__init__(None)

        # store input
        self.dataset = dataset
        self.balanced_class_name = balanced_class_name
        self.num_balanced_classes = num_balanced_classes
        self.batch_size = batch_size
        self.balanced_class_weights = balanced_class_weights
        self.balanced_class_probs = balanced_class_probs
        self.num_batches = num_batches
        self.use_dataset_cache = use_dataset_cache

        # validate input
        if balanced_class_weights is not None and balanced_class_probs is not None:
            raise Exception('Set either balanced_class_weights or balanced_class_probs, not both.')
        elif balanced_class_weights is None and balanced_class_probs is None:
            if batch_size % num_balanced_classes != 0:
                raise Exception(f'batch_size ({batch_size}) % num_balanced_classes ({num_balanced_classes}) must be 0')
        elif balanced_class_weights is not None:
            if len(balanced_class_weights) != num_balanced_classes:
                raise Exception(
                    f'Expecting balance_class_weights ({balanced_class_weights}) to have a weight per balanced class ({num_balanced_classes})')
            if sum(balanced_class_weights) != batch_size:
                raise Exception(f'balanced_class_weights {balanced_class_weights} expected to sum up to batch_size {batch_size}')
        else:
            # noinspection PyTypeChecker
            if len(balanced_class_probs) != num_balanced_classes:
                raise Exception(
                    f'Expecting balance_class_probs ({balanced_class_probs}) to have a probability per balanced class ({num_balanced_classes})')
            if not math.isclose(sum(balanced_class_probs), 1.0):
                raise Exception(f'balanced_class_probs {balanced_class_probs} expected to sum up to 1.0')

        # if weights not specified, set weights to equally balance per batch
        if self.balanced_class_weights is None and self.balanced_class_probs is None:
            self.balanced_class_weights = [self.batch_size // self.num_balanced_classes] * self.num_balanced_classes

        lgr = logging.getLogger('Fuse')
        lgr.debug(f'FuseSamplerBalancedBatch: balancing per batch - balanced_class_name {self.balanced_class_name}, '
                  f'batch_size={batch_size}, weights={self.balanced_class_weights}, probs={self.balanced_class_probs}')

        # get balanced classes per each sample
        self.balanced_classes = dataset.get(None, self.balanced_class_name, use_cache=use_dataset_cache)
        self.balanced_classes = np.array(self.balanced_classes)
        self.balanced_class_indices = [np.where(self.balanced_classes == cls_i)[0] for cls_i in range(self.num_balanced_classes)]
        self.balanced_class_sizes = [len(self.balanced_class_indices[cls_i]) for cls_i in range(self.num_balanced_classes)]
        lgr.debug('FuseSamplerBalancedBatch: samples per each balanced class {}'.format(self.balanced_class_sizes))

        # debug - simple batch
        batch_mode = FuseUtilsDebug().get_setting('sampler_batch_mode')
        if batch_mode == 'simple':
            num_avail_bcls = 0
            for bcls_num_samples in self.balanced_class_sizes:
                if bcls_num_samples != 0:
                    num_avail_bcls += 1

            self.balanced_class_weights = None
            self.balanced_class_probs = [1.0/num_avail_bcls if bcls_num_samples != 0 else 0.0 for bcls_num_samples in self.balanced_class_sizes]
            lgr.info('FuseSamplerBalancedBatch: debug mode - override to random sample')

        # calc batch index to balanced class mapping according to weights
        if self.balanced_class_weights is not None:
            self.batch_index_to_class = []
            for balanced_cls in range(self.num_balanced_classes):
                self.batch_index_to_class.extend([balanced_cls] * self.balanced_class_weights[balanced_cls])
        else:
            # probabilistic method - will be randomly select per epoch
            self.batch_index_to_class = None

        # make sure that size != 0 for all balanced classes
        for cls_size in enumerate(self.balanced_class_sizes):
            if (self.balanced_class_weights is not None and self.balanced_class_weights != 0) or (
                    self.balanced_class_probs is not None and self.balanced_class_probs != 0.0):
                if cls_size == 0:
                    msg = f'Every balanced class must include at least one sample (num of samples per balanced class{self.balanced_class_sizes})'
                    raise Exception(msg)

        # Shuffle balanced class indices
        for indices in self.balanced_class_indices:
            np.random.shuffle(indices)

        # Calculate num batches. Number of batches to iterate over all data at least once
        # Calculate only if not directly specified by the user
        if self.num_batches is None:
            if self.balanced_class_weights is not None:
                balanced_class_weighted_sizes = [self.balanced_class_sizes[cls_i] // self.balanced_class_weights[cls_i] if self.balanced_class_weights[cls_i] != 0 else 0 for cls_i in
                                                 range(self.num_balanced_classes)]
            else:
                # approximate size!
                balanced_class_weighted_sizes = [
                    self.balanced_class_sizes[cls_i] // (self.balanced_class_probs[cls_i] * self.batch_size) if self.balanced_class_probs[
                                                                                                                    cls_i] != 0.0 else 0 for
                    cls_i in range(self.num_balanced_classes)]
            bigger_balanced_class_weighted_size = max(balanced_class_weighted_sizes)
            self.num_batches = int(bigger_balanced_class_weighted_size) + 1
            lgr.debug(f'FuseSamplerBalancedBatch: num_batches = {self.num_batches}')

        # pointers per class
        self.cls_pointers = [0] * self.num_balanced_classes
        self.sample_pointer = 0

    def __iter__(self) -> np.ndarray:
        for batch_idx in range(self.num_batches):
            yield self._make_batch()

    def __len__(self) -> int:
        return self.num_batches

    def _get_sample(self, balanced_class: int) -> Any:
        """
        sample index given balanced class value
        :param balanced_class: integer representing balanced class value
        :return: sample index
        """
        if self.balanced_class_indices[balanced_class].shape[0] == 0:
            msg = f'There are no samples in balanced class {balanced_class}'
            logging.getLogger('Fuse').error(msg)
            raise Exception(msg)

        sample_idx = self.balanced_class_indices[balanced_class][self.cls_pointers[balanced_class]]

        self.cls_pointers[balanced_class] += 1
        if self.cls_pointers[balanced_class] == self.balanced_class_sizes[balanced_class]:
            self.cls_pointers[balanced_class] = 0
            np.random.shuffle(self.balanced_class_indices[balanced_class])

        return sample_idx

    def _make_batch(self) -> list:
        """
        :return: list of indices to collate batch
        """
        if self.batch_index_to_class is not None:
            batch_index_to_class = self.batch_index_to_class
        else:
            # calc one according to probabilities
            batch_index_to_class = np.random.choice(np.arange(self.num_balanced_classes), self.batch_size, p=self.balanced_class_probs)
        batch_sample_indices = []
        for batch_index in range(self.batch_size):
            balanced_class = batch_index_to_class[batch_index]
            batch_sample_indices.append(self._get_sample(balanced_class))

        np.random.shuffle(batch_sample_indices)
        return batch_sample_indices
