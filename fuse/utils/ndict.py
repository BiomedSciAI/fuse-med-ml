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
from __future__ import annotations
from _collections_abc import dict_items, dict_keys

import copy
import types
import numpy
import torch
from typing import Any, Callable, Iterator, Optional, Sequence, Union, List, MutableMapping


class NDict(dict):
    """
    (Lazy) N(ested)Dict - wraps a python dict, and allows to access "nested" elements via '.' separated key desc

    NOTE: assumes that all keys (including nested) are:
        1. strings
        2. do not contain '.' within a single key, as '.' is used as a special symbol for accessing deeper level nested dict.

    For example:

    nx = NDict(x)
    nx['a.b'] = 14
    assert nx['a.b'] == 14

    if the result is a non-leaf, you will get a NDict instance, for example
    assert nx['a']['d.zz'] == 'abc'
    """

    def __init__(
        self, dict_like: Union[dict, tuple, types.GeneratorType, NDict, None] = None, already_flat: bool = False
    ):
        """
        :param dict_like: the data with which to populate the nested dictionary, in case of NDict it acts as view constructor,
            otherwise we just set all the keys and values using the setitem function
        :param already_flat: optimization option. set to True only if you are sure that the input "dict_like" is a dict without nested
        """

        if dict_like is None:
            self._stored = dict()

        elif isinstance(dict_like, NDict):
            self._stored = dict_like._stored

        elif isinstance(dict_like, dict) and already_flat:
            self._stored = dict_like

        elif dict_like is not None:
            self._stored = dict()
            if not isinstance(dict_like, MutableMapping):
                dict_like = dict(dict_like)
            for k, d in dict_like.items():
                self[k] = d

        else:
            raise Exception(f"Not supported dict_like type: {type(dict_like)}")

    def to_dict(self) -> dict:  # OPT
        """
        converts to standard python dict
        """
        return self._stored

    def clone(self, deepcopy: bool = True) -> NDict:  # OPT
        """
        does a deep or a shallow copy, shallow copy means only top level keys are copied and the values are only referenced
        in deep copy, all values are copied recursively

        :param deepcopy: if true, does deep copy, otherwise does a shallow copy
        """
        if not deepcopy:
            return NDict(copy.copy(self._stored))
        else:
            return NDict(copy.deepcopy(self._stored))

    def flatten(self) -> dict:  # OPT
        """
        Legacy from previous implementation of NDict where we store the data with nested dictionaries.
        currently we store the data with one flat dictionary
        """
        return self._stored

    def keypaths(self) -> List[str]:
        """
        :return: a list of keypaths (i.e. "a.b.c.d") to all values in the nested dict
        """
        return list(self._stored.keys())

    def keys(self) -> dict_keys:
        """
        returns the top-level keys of the dictionary
        """
        return list(self._stored.keys())

    def values(self) -> dict_items:
        return self._stored.values()

    def items(self) -> dict_items:
        return self._stored.items()

    def merge(self, other: dict) -> NDict:
        """
        inplace merge between self and other.

        :param other: the dic
        """
        for k, v in other.items():
            self[k] = v

        return self

    def __getitem__(self, key: str) -> Any:
        """
        "traverses" the nested dict by the path extracted from splitting the key on '.', if key not found,
        optionally shows the possible closest options.
        if a prefix is given (the key exists but it is not a leaf), returns the corresponding sub dictionary

        :param key: dot delimited keypath into the nested dict
        """
        # the key is a full key for a value
        if key in self._stored:
            return self._stored[key]

        # the key is a prefix for other value(s)
        elif self.is_prefix(key):  # TODO can be more optimized. we pass here once and in the "get_sub_dict" once again
            # collect "sub-dict"
            return self.get_sub_dict(key)

        else:
            raise NestedKeyError(key, self)

    def is_prefix(self, key: str) -> bool:
        """
        return True iff the key is a *strictly* prefix of another key(s) in the NDict.
        :param key:

        Example:
            >>> ndict = NDict()
            >>> ndict["a.b.c"] = "d"
            >>> ndict.is_prefix("a")
            True
            >>> ndict.is_prefix("a.b")
            True
            >>> ndict.is_prefix("a.b.c")
            False    # STRICTLY PREFIX!
        """
        # iterate over all keys and looking for a match
        for kk in self.keypaths():
            if kk.startswith(f"{key}."):
                return True

        # a match wasn't found
        return False

    def get_sub_dict(self, key: str) -> NDict:
        """
        TODO

        Example:
            >>> ndict = NDict()
            >>> ndict["a.b.c"] = "x"
            >>> ndict["a.b.d"] = "y"
            >>> ndict["a.b.e"] = "z"
            >>> ndict.get_sub_dict("a.b")
            {'c': 'x', 'd': 'y', 'e': 'z'}
        """
        res = NDict()
        key = key + "."
        for kk in self.keypaths():
            if kk.startswith(key):
                sub_key = kk.replace(key, "", 1)
                res[sub_key] = self[kk]

        return res

    def __setitem__(self, key: str, value: Any) -> None:
        """
        go over the the dictionary according to the path, create the nodes that does not exist
        :param key: the keypath
        :param value: value to set
        """
        # if value is dictionary add to self key by key to avoid from keys with delimiter "."
        if isinstance(value, MutableMapping):
            for sub_key in value:
                self[f"{key}.{sub_key}"] = value[sub_key]
            return
        self._stored[key] = value

    def __delitem__(self, key: str) -> None:  # OPT
        """
        :param key:
        TODO should we delete both value and prefix ?
        """
        # delete specific (key, value)
        if key in self._stored:
            del self._stored[key]

        # delete entire branch
        elif self.is_prefix(key):
            for kk in self.keypaths():
                if kk.startswith(f"{key}."):
                    del self[kk]

        else:
            raise NestedKeyError(key, self)

    def get_closest_key(self, key: str) -> str:
        """
        For a given keypath, returns the longest valid keypath in the current nested dict
        :param key: a full keypath with dot delimiter
        """
        if key in self._stored:
            return key

        key_parts = key.split(".")
        for i in range(len(key_parts)):
            if self.is_prefix(key):
                return key
            key = ".".join(key_parts[:-i])

        return ""

    def pop(self, key: str) -> Any:  # OPT
        """
        return the value nested_dict[key] and remove the key from the dict.

        :param nested_dict: the dictionary
        :param key: the key to return and remove
        """
        res = self[key]
        del self[key]
        return res

    def indices(self, indices: numpy.ndarray) -> dict:
        """
        Extract the specified indices from each element in the dictionary (if possible)

        :param nested_dict: input dict
        :param indices: indices to extract. Either list of indices or boolean numpy array
        :return: NDict with only the required indices
        """
        new_dict = {}
        all_keys = self.keypaths()
        for key in all_keys:
            try:
                value = self[key]
                if isinstance(value, (numpy.ndarray, torch.Tensor)):
                    new_value = value[indices]
                elif isinstance(value, Sequence):
                    new_value = [item for i, item in enumerate(value) if indices[i]]
                else:
                    new_value = value
                new_dict[key] = new_value
            except:
                print(f"failed to process key {key}")
                raise
        return new_dict

    def apply_on_all(self, apply_func: Callable, *args: Any) -> None:
        """
        Inplace apply specified function on all the dictionary's values

        :param apply_func: function to apply
        :param args: custom arguments for the 'apply_func' function
        """
        all_keys = self.keypaths()
        for key in all_keys:
            new_value = apply_func(self[key], *args)
            self[key] = new_value

    def __reduce__(self) -> Union[str, tuple]:
        return super().__reduce__()

    def __iter__(self) -> Iterator:
        return iter(self._stored)

    def __len__(self) -> int:
        return len(self._stored)

    def __str__(self) -> str:
        return str(self._stored)

    def __repr__(self) -> str:
        return repr(self._stored)

    def __contains__(self, o: str) -> bool:
        return o == self.get_closest_key(o)

    def get(self, key: str, default_value: Any = None) -> Any:
        if key not in self:
            return default_value
        return self[key]

    def get_multi(self, keys: Optional[List[str]] = None) -> NDict:
        """
        get multiple

        :param keys:
        """
        if keys is None:
            keys = self.keypaths()  # take all keys

        ans = NDict()

        for k in keys:
            curr = self[k]
            ans[k] = curr

        return ans

    def print_tree(self, print_values: bool = False) -> None:
        """
        print the inner structure of the nested dict with a tree-like structure.

        :param print_values: set to True in order to also print ndict's stored values


        Example:

            >>> ndict = NDict()
            >>> ndict["data.input.drug"] = "this_is_a_drug_seq"
            >>> ndict["data.input.target"] = "this_is_a_target_seq"
            >>>
            >>> ndict.print_tree()
            --- data
            ------ input
            --------- drug
            --------- target
            >>>
            >>> ndict.print_tree(print_values=True)
            --- data
            ------ input
            --------- drug -> this_is_a_drug_seq
            --------- target -> this_is_a_target_seq

        """
        self._print_tree_static(self._stored, print_values=print_values)

    @staticmethod
    def _print_tree_static(data_dict: dict, level: int = 0, print_values: bool = False) -> None:
        """
        static-method to print the inner structure of a dict in a tree-like structure.

        :param level: current recursive level inside the ndict
        :param print_values: set to True in order to also print ndict's stored values
        """
        keys = data_dict.keys()
        level += 1
        for key in keys:
            if type(data_dict[key]) == dict:
                print("---" * level, key)
                NDict._print_tree_static(data_dict[key], level, print_values=print_values)
            else:
                if print_values:
                    print("---" * level, key, "->", data_dict[key])
                else:
                    print("---" * level, key)

    def describe(self) -> None:
        for k in self.keypaths():
            print(f"{k}")
            val = self[k]
            print(f"\ttype={type(val)}")
            if hasattr(val, "shape"):
                print(f"\tshape={val.shape}")


class NestedKeyError(KeyError):
    def __init__(self, key: str, d: NDict) -> None:
        partial_key = d.get_closest_key(key)
        if partial_key == "":
            error_str = f"Error: key {key} does not exist\n. All keys: {d.keypaths()}"
        else:
            partial_ndict = d[partial_key]

            if isinstance(partial_ndict, NDict):
                options = str([f"{partial_key}.{k}" for k in partial_ndict.keypaths()])
                error_str = f"Error: key {key} does not exist\n. Possible keys on the same branch are: {options}. All keys {d.keypaths()}"
            else:
                error_str = (
                    f"Error: key {key} does not exist\n. Closest key is: {partial_key}. All keys: {d.keypaths()}"
                )
        print(error_str)
        super().__init__(error_str)
