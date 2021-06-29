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

from typing import Any, Set, Callable, Optional, List, Sequence, Union

import numpy
import torch


class FuseUtilsHierarchicalDict:
    @classmethod
    def get(cls, hierarchical_dict: dict, key: str):
        """
        get(dict, 'x.y.z') <==> dict['x']['y']['z']
        """
        # split according to '.'
        hierarchical_key = key.split('.')

        # go over the the dictionary towards the requested value
        try:
            value = hierarchical_dict[hierarchical_key[0]]
            for sub_key in hierarchical_key[1:]:
                value = value[sub_key]
            return value
        except:
            flat_dict = FuseUtilsHierarchicalDict.flatten(hierarchical_dict)
            if key in flat_dict:
                return flat_dict[key]
            else:
                raise KeyError(f'key {key} does not exist\n. Possible keys are: {str(list(flat_dict.keys()))}')

    @classmethod
    def set(cls, hierarchical_dict: dict, key: str, value: Any) -> None:
        """
        set(dict, 'x.y.z', value) <==> dict['x']['y']['z'] = value
        If either 'x', 'y' or 'z' nodes do not exist, this function will create them
        """
        # split according to '.'
        hierarchical_key = key.split('.')

        # go over the the dictionary according to the path, create the nodes that does not exist
        element = hierarchical_dict
        for key in hierarchical_key[:-1]:
            if key not in element:
                element[key] = {}
            element = element[key]

        # set the value
        element[hierarchical_key[-1]] = value

    @classmethod
    def get_all_keys(cls, hierarchical_dict: dict, include_values: bool = False) -> Union[List[str], dict]:
        """
        Get all hierarchical keys in  hierarchical_dict
        """
        all_keys = {}
        for key in hierarchical_dict:
            if isinstance(hierarchical_dict[key], dict):
                all_sub_keys = FuseUtilsHierarchicalDict.get_all_keys(hierarchical_dict[key], include_values=True)
                keys_to_add = {f'{key}.{sub_key}':all_sub_keys[sub_key] for sub_key in all_sub_keys}
                all_keys.update(keys_to_add)
            else:
                all_keys[key] = hierarchical_dict[key]
        if include_values:
            return all_keys
        else:
            return list(all_keys.keys())

    @classmethod
    def subkey(cls, key: str, start: int, end: Optional[int]) -> Optional[str]:
        """
        Sub string of hierarchical key.
        Example: subkey('a.b.c.d.f', 1, 3) -> 'b.c'
        :param key: the original key
        :param start: start index
        :param end: end index, not including
        :return: str
        """
        key_parts = key.split('.')

        # if end not specified set to max.
        if end is None:
            end = len(key_parts)

        if len(key_parts) < start or len(key_parts) < end:
            return None

        res = '.'.join(key_parts[start:end])
        return res

    @classmethod
    def apply_on_all(cls, hierarchical_dict: dict, apply_func: Callable, *args: Any) -> None:
        all_keys = cls.get_all_keys(hierarchical_dict)
        for key in all_keys:
            new_value = apply_func(cls.get(hierarchical_dict, key), *args)
            cls.set(hierarchical_dict, key, new_value)
        pass

    @classmethod
    def flatten(cls, hierarchical_dict: dict) -> dict:
        """
        Flatten the dict
        @param hierarchical_dict: dict to flatten
        @return: dict where keys are the hierarchical_dict keys separated by periods.
        """
        flat_dict = {}
        return cls.get_all_keys(hierarchical_dict, include_values=True)

    @classmethod
    def indices(cls, hierarchical_dict: dict, indices: List[int]) -> dict:
        """
        Extract the specified indices from each element in the dictionary (if possible)
        :param hierarchical_dict: input dict
        :param indices: indices to extract
        :return: dict with only the required indices
        """
        new_dict = {}
        all_keys = cls.get_all_keys(hierarchical_dict)
        for key in all_keys:
            value = cls.get(hierarchical_dict, key)
            if isinstance(value, numpy.ndarray) or isinstance(value, torch.Tensor):
                new_value = value[indices]
            elif isinstance(value, Sequence):
                new_value =[item for i, item in enumerate(value) if indices[i]]
            else:
                new_value = value
            cls.set(new_dict, key, new_value)
        return new_dict

    @classmethod
    def to_string(cls, hierarchical_dict: dict) -> str:
        """
        Get flat string including thr content of the dictionary
        :param hierarchical_dict: input dict
        :return: string
        """
        keys = cls.get_all_keys(hierarchical_dict)
        keys = sorted(keys)
        res = ''
        for key in keys:
            res += f'{key} = {FuseUtilsHierarchicalDict.get(hierarchical_dict, key)}\n'

        return res

    @classmethod
    def pop(cls, hierarchical_dict: dict, key:str):
        """
        return the value hierarchical_dict[key] and remove the key from the dict.
        :param hierarchical_dict: the dictionary
        :param key: the key to return and remove
        """
        # split according to '.'
        hierarchical_key = key.split('.')
        # go over the the dictionary towards the requested value
        try:
            key_idx = len(hierarchical_key) - 1
            value = hierarchical_dict[hierarchical_key[0]] if key_idx > 0 else hierarchical_dict
            for sub_key in hierarchical_key[1:-1]:
                value = value[sub_key]
            return value.pop(hierarchical_key[key_idx])
        except:
            flat_dict = FuseUtilsHierarchicalDict.flatten(hierarchical_dict)
            if key in flat_dict:
                return flat_dict[key]
            else:
                raise KeyError(f'key {key} does not exist\n. Possible keys are: {str(list(flat_dict.keys()))}')

    @classmethod
    def is_in(cls, hierarchical_dict: dict, key:str) -> bool:
        """
        Returns True if the full key is in dict, False otherwise.
        e.g., for dict = {'a':1, 'b.c':2} is_in(dict, 'b.c') returns True, but is_in(dict, 'c') returns False.

        :param hierarchical_dict: dict to check
        :param key: key to search
        :return: key in hierarchical_dict
        """
        return key in cls.get_all_keys(hierarchical_dict)