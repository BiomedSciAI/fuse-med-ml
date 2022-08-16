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
from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Union
from fuse.data.ops.op_base import OpReversibleBase
import numpy as np

from fuse.data import OpBase
import torch
from torch import Tensor
from fuse.utils.ndict import NDict


class Cast:
    """
    Cast methods
    """

    @staticmethod
    def to_tensor(value: Any, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> Tensor:
        """
        Convert many types to tensor
        """
        if isinstance(value, torch.Tensor) and dtype is None and device is None:
            pass  # do nothing
        elif isinstance(value, (torch.Tensor)):
            value = value.to(dtype=dtype, device=device)
        elif isinstance(value, (np.ndarray, int, float, list)):
            value = torch.tensor(value, dtype=dtype, device=device)
        else:
            raise Exception(f"Unsupported type {type(value)} - add here support for this type")

        return value

    @staticmethod
    def to_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """
        Convert many types to numpy
        """
        if isinstance(value, np.ndarray) and dtype is None:
            pass  # do nothing
        elif isinstance(value, (torch.Tensor, int, float, list, np.ndarray)):
            value = np.array(value, dtype=dtype)
        else:
            raise Exception(f"Unsupported type {type(value)} - add here support for this type")

        return value

    @staticmethod
    def to_int(value: Any) -> np.ndarray:
        """
        Convert many types to int
        """
        if isinstance(value, int):
            pass  # do nothing
        elif isinstance(value, (torch.Tensor, np.ndarray, float, str, bytes)):
            value = int(value)
        else:
            raise Exception(f"Unsupported type {type(value)} - add here support for this type")

        return value

    @staticmethod
    def to_float(value: Any) -> np.ndarray:
        """
        Convert many types to float
        """

        if isinstance(value, float):
            pass  # do nothing
        elif isinstance(value, (torch.Tensor, np.ndarray, int, str)):
            value = float(value)
        else:
            raise Exception(f"Unsupported type {type(value)} - add here support for this type")

        return value

    @staticmethod
    def to_list(value: Any) -> np.ndarray:
        """
        Convert many types to list
        """

        if isinstance(value, list):
            pass  # do nothing
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            value = value.tolist()
        else:
            raise Exception(f"Unsupported type {type(value)} - add here support for this type")

        return value

    @staticmethod
    def to(value: Any, type_name: str, **kwargs) -> Any:
        """
        Convert any type to type specified in type_name
        """

        if type_name == "ndarray":
            return Cast.to_numpy(value, **kwargs)
        if type_name == "Tensor":
            return Cast.to_tensor(value, **kwargs)
        if type_name == "float":
            return Cast.to_float(value, **kwargs)
        if type_name == "int":
            return Cast.to_int(value, **kwargs)
        if type_name == "list":
            return Cast.to_list(value, **kwargs)

    @staticmethod
    def like(value: Any, like_value: Any) -> Any:
        """
        Convert any type to type specified in type_name
        """

        if isinstance(like_value, np.ndarray):
            return Cast.to_numpy(value, dtype=like_value.dtype)
        if isinstance(like_value, torch.Tensor):
            return Cast.to_tensor(value, dtype=like_value.dtype, device=like_value.device)
        if isinstance(like_value, float):
            return Cast.to_float(value)
        if isinstance(like_value, int):
            return Cast.to_int(value)
        if isinstance(like_value, list):
            return Cast.to_list(value)


class OpCast(OpReversibleBase):
    def __call__(
        self, sample_dict: NDict, op_id: Optional[str], key: Union[str, Sequence[str]], **kwargs
    ) -> Union[None, dict, List[dict]]:
        """
        See super class
        :param key: single key or list of keys from sample_dict to convert
        """
        if isinstance(key, str):
            keys = [key]
        else:
            keys = key

        for key_name in keys:
            value = sample_dict[key_name]
            sample_dict[f"{op_id}_{key_name}"] = type(value).__name__
            value = self._cast(value, **kwargs)
            sample_dict[key_name] = value

        return sample_dict

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        type_name = sample_dict[f"{op_id}_{key_to_follow}"]
        value = sample_dict[key_to_reverse]
        value = Cast.to(value, type_name)
        sample_dict[key_to_reverse] = value

        return sample_dict

    @abstractmethod
    def _cast(self):
        raise NotImplementedError


class OpToTensor(OpCast):
    """
    Convert many types to tensor
    """

    def _cast(self, value: Any, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> Tensor:
        return Cast.to_tensor(value, dtype, device)


class OpToNumpy(OpCast):
    """
    Convert many types to numpy
    """

    def _cast(self, value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:

        return Cast.to_numpy(value, dtype)


class OpToInt(OpCast):
    """
    Convert many types to int
    """

    def _cast(self, value: Any) -> int:
        return Cast.to_int(value)


class OpToFloat(OpCast):
    """
    Convert many types to float
    """

    def _cast(self, value: Any) -> float:
        return Cast.to_float(value)


class OpOneHotToNumber(OpBase):
    """
    Convert one-hot encoding into numbers.
    for example:
        [1, 0, 0, 0] -> 0
        [0, 0, 1, 0] -> 2
    """

    def __init__(self, num_classes, verify_arguments: bool = True):
        """
        :param num_classes: Number of class the Op should expect.
        :param verify_arguments: Defualt is True - can be set to False for a speedup.
        """
        super().__init__()
        self._num_classes = num_classes
        self._verify_arguments = verify_arguments

    def __call__(self, sample_dict: NDict, key: str, **kwargs) -> Union[None, dict, List[dict]]:
        """
        :param key: the sample_dict's key where the one-hot vector is located.
                    The corresponding number will be save in the same key (instead of the one-hot)
        """
        one_hot_vector = np.array(sample_dict[key])

        if self._verify_arguments:
            assert self._num_classes == len(
                one_hot_vector
            ), f"Error: One-hot vector length is {len(one_hot_vector)} but number of classes is {self._num_classes}"
            assert self.is_one_hot_vector(one_hot_vector), f"expect one-hot vector, instead got: {one_hot_vector} ."

        value = one_hot_vector.argmax()
        sample_dict[key] = torch.tensor(value)

        return sample_dict

    def is_one_hot_vector(self, vector: np.ndarray) -> bool:
        """
        Return true iff vector is one-hot with unique class. (not multiple features)

        :param vector:
        """

        max_value = vector.max()
        min_value = vector.min()
        sum_value = vector.sum()

        ans = (max_value == 1) and (min_value == 0) and (sum_value == 1)
        return ans
