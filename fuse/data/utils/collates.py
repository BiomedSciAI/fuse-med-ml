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
from typing import Any, Callable, Dict, List, Sequence, Optional, Tuple

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
        special_handlers_keys: Optional[Dict[str, Callable]] = None,
        post_collate_special_handlers_keys: Optional[List[Callable]] = None,
        add_to_batch_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        :param skip_keys: do not collect the listed keys
        :param keep_keys: specifies a list of keys to collect. See raise_error_key_missing argument (dealing with missing keys).
        :param special_handlers_keys: per key specify a callable which gets as an input list of values and convert it to a batch.
                                      The rest of the keys will be converted to batch using PyTorch default collate_fn()
                                      Example of such Callable can be seen in the CollateDefault.pad_all_tensors_to_same_size.
        :param post_collate_special_handlers_keys: specify a callable which gets the batch_dict as an input and applies post processing to the batched tensors.
        :param raise_error_key_missing: if False, will not raise an error if there are keys that do not exist in some of the samples. Instead will set those values to None.
        :param add_to_batch_dict: optional, fixed items to add to batch_dict
        """
        super().__init__(skip_keys, raise_error_key_missing)
        self._special_handlers_keys = {}
        if special_handlers_keys is not None:
            self._special_handlers_keys.update(special_handlers_keys)
        self._special_handlers_keys[
            get_sample_id_key()
        ] = CollateDefault.just_collect_to_list
        self._keep_keys = keep_keys
        self._add_to_batch_dict = add_to_batch_dict

        self._post_collate_special_handlers_keys = post_collate_special_handlers_keys

    def __call__(self, samples: List[Dict]) -> Dict:
        """
        collate list of samples into batch_dict
        :param samples: list of samples
        :return: batch_dict
        """
        batch_dict = NDict()

        # collect all keys
        if self._keep_keys:
            keys = self._keep_keys
        else:
            keys = self._collect_all_keys(samples)

        # collect values
        for key in keys:
            # skip keys
            if key in self._skip_keys:
                continue

            try:
                # collect values into a list
                (
                    collected_values,
                    has_error,
                    has_missing_values,
                ) = self._collect_values_to_list(samples, key)

                # batch values
                self._batch_dispatch(
                    batch_dict,
                    samples,
                    key,
                    has_error or has_missing_values,
                    collected_values,
                )
            except:
                print(f"Error: Failed to collect key {key}")
                raise

        if self._add_to_batch_dict is not None:
            batch_dict.update(self._add_to_batch_dict)

        if self._post_collate_special_handlers_keys is not None:
            for callable in self._post_collate_special_handlers_keys:
                callable(batch_dict)

        return batch_dict

    def _batch_dispatch(
        self,
        batch_dict: dict,
        samples: List[dict],
        key: str,
        has_error: bool,
        collected_values: list,
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
        elif isinstance(
            collected_values[0],
            (torch.Tensor, np.ndarray, float, int, str, bytes),
        ):
            # batch with default PyTorch implementation
            batch_dict[key] = default_collate(collected_values)
        else:
            batch_dict[key] = collected_values

    @staticmethod
    def just_collect_to_list(values: List[Any]) -> List[Any]:
        """
        special handler doing nothing - will just keep the collected list
        """
        return values

    @staticmethod
    def pad_all_tensors_to_same_size(
        values: List[torch.Tensor],
        pad_val: float = 0.0,
        min_size_per_dim: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """
        pad tensors and create a batch - the shape will be the max size per dim
        values: list of tensor - all should have the same number of dimensions
        pad_val: constant value for padding
        min_size_per_dim: defines, per dimension, the minimal size in the post-collated tensor (excluding the batch dimension, which you shouldn't provide)
            this can be useful to prevent OOM due to memory fragmentation
            for example, you can use min_size_per_dim = (1000, -1) to make sure that the final collated tensor first dimension is at least 1000

        :return: torch.stack of padded tensors
        """

        # verify all are tensor and that they have the same dim size
        assert isinstance(
            values[0], torch.Tensor
        ), f"Expecting just tensors, got {type(values[0])}"
        num_dims = len(values[0].shape)

        for value in values:
            assert isinstance(
                value, torch.Tensor
            ), f"Expecting just tensors, got {type(value)}"
            assert (
                len(value.shape) == num_dims
            ), f"Expecting all tensors to have the same dim size, got {len(value.shape)} and {num_dims}"

        # get max per dim - this is the ovserved actual max from the batch
        max_per_dim = np.amax(np.stack([value.shape for value in values]), axis=0)

        if min_size_per_dim is not None:
            assert isinstance(min_size_per_dim, tuple)
            if len(min_size_per_dim) < len(max_per_dim):
                raise Exception(
                    f" length for min_size_per_dim={min_size_per_dim} expected to be >= max_per_dim={max_per_dim} but found length {len(min_size_per_dim)} and {len(max_per_dim)} respectively ! "
                )
            assert all(
                [(x > 0) or (x == -1) for x in min_size_per_dim]
            ), "allowed values for elements in min_size_per_dim are only positive integer or -1"
            max_per_dim = [
                max(actual, minimal_requested) if minimal_requested != -1 else actual
                for (actual, minimal_requested) in zip(max_per_dim, min_size_per_dim)
            ]

        # pad
        def _pad_size(value: torch.Tensor, dim: int) -> List[int]:
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

    @staticmethod
    def crop_padding(
        input_ids_list: List[torch.Tensor], pad_token_id: int
    ) -> torch.Tensor:
        """
        Crop padding of a batch of input_ids 1D tensors to the minimum length possible.

        Args:
            input_ids_list (list of torch.Tensor): List of input_ids tensors, where each tensor represents a sequence.
            pad_token_id (int): ID of the padding token used in input_ids tensors.

        Returns:
            torch.Tensor: Batched and cropped input_ids tensor with padding removed to the maximum length.

        Example:
            >>> input_ids_list = [
            ...     torch.tensor([101, 2054, 2003, 0, 0, 0, 0, 0, 0, 0]),
            ...     torch.tensor([101, 2023, 2003, 1037, 1999, 0, 0, 0, 0, 0]),
            ...     torch.tensor([101, 2002, 0, 0, 0, 0, 0, 0, 0, 0]),
            ... ]
            >>> pad_token_id = 0
            >>> cropped_batch = crop_padding_to_max_length(input_ids_list, pad_token_id)
            >>> print(cropped_batch)
            tensor([[ 101, 2054, 2003, 0,    0],
                    [ 101, 2023, 2003, 1037, 1999],
                    [ 101, 2002, 0,    0,    0]])

        Note:
            This function assumes that the input_ids tensors are already padded, and it crops the sequences
            to the minimum length by removing trailing padding tokens.
        """
        min_length = max(
            len(ids) - (ids == pad_token_id).sum().item() for ids in input_ids_list
        )
        cropped_sequences = [ids[:min_length] for ids in input_ids_list]
        batched_sequences = torch.stack(cropped_sequences, dim=0)
        return batched_sequences

    @staticmethod
    def crop_length_to_match_target_key(
        batch_dict: dict, target_key: str, keys_to_match: List[str]
    ) -> None:
        """
        Match the 1st dimension (typically the sequence length) of tensors in batch_dict, specified by keys, to a target tensor.

        Args:
            :param batch_dict: input dictionary with batched tensors with expected shape (B, L, *) where B is the batch dimension and L is a sequential dimension. L may vary across the elements in the batch_dict. * is any number of additional dimensions.
            :param target_key: key in batch_dict of the tensors with desired sequential dimension, L'.
            :param keys_to_match: list of keys in batch_dict with tensors who's 1st dimension should be converted to match the target_key tensor. The tensors to match should have 1st dimenions L >= L'.

        Returns:
            None, elements are modified in the batch_dict.
        """
        target_length = batch_dict[target_key].shape[1]
        for key in keys_to_match:
            batch_dict[key] = batch_dict[key][:, :target_length]
