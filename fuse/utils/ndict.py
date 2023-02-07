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
    """N(ested)Dict - wraps a python dict, and allows to access nested elements via '.' separated key desc

    NOTE: assumes that all keys (including nested) are:
        1. strings
        2. do not contain '.' within a single key, as '.' is used as a special symbol for accessing deeper level nested dict.

    For example:

    x = dict(
        a = dict(
            b = 10,
            c = 12,
            d = dict(
                zz = 'abc',
            )
        ),
        c = 100,
    )

    nx = NDict(x)
    nx['a.b'] = 14
    assert nx['a.b'] == 14

    if the result is a non-leaf, you will get a NDict instance, for example
    assert nx['a']['d.zz'] == 'abc'

    In addition to standard python dict methods, implements:
    * flatten
    * to_dict
    * combine

    """

    def __init__(self, dict_like: Union[dict, tuple, types.GeneratorType, NDict, None] = None):
        """
        :param dict_like: the data with which to populate the nested dictionary, in case of NDict it acts as view constructor,
            otherwise we just set all the keys and values using the setitem function
        """

        self._stored = dict()

        if dict_like is None:
            self._stored = {}
        elif isinstance(dict_like, NDict):
            self._stored = dict_like._stored
        else:
            if not isinstance(dict_like, MutableMapping):
                dict_like = dict(dict_like)
            for k, d in dict_like.items():
                self[k] = d

    # NDict custom methods
    def to_dict(self) -> dict:  # OPT
        """
        converts to standard python dict
        """
        return self._stored

    def clone(self, deepcopy: bool = True) -> NDict:  # OPT
        """
        does a deep or a shallow copy, shallow copy means only top level keys are copied and the values are only referenced
        in deep copy, all values are copied recursively
        :param deepcopy: if true, does deep copy, otherwise does shallow copy
        """
        if not deepcopy:
            return NDict(copy.copy(self._stored))
        else:
            return NDict(copy.deepcopy(self._stored))

    def flatten(self) -> dict:  # OPT
        """
        flattens the dictionary
        :returns dict

        For example:

        nx = NDict({'a': {'b': 14, 'c': 12}, 'c': 100, 'z': {'foo': {'boo': 111}}})
        print(nx.flatten())
        {'a.b': 14, 'a.c': 12, 'c': 100, 'z.foo.boo': 111}

        #you can use it to get a list of the flat keys:
        print(nx.flatten().keys())
        """

        return self._stored  # OPT store it flat

    # @staticmethod
    # def _flatten_static(item: Union[dict, Any], prefix: str, flat_dict: dict) -> None:
    #     if isinstance(item, MutableMapping):
    #         for key, value in item.items():
    #             if prefix is None:
    #                 cur_prefix = key
    #             else:
    #                 cur_prefix = f"{prefix}.{key}"
    #             NDict._flatten_static(value, cur_prefix, flat_dict)
    #     else:
    #         flat_dict[prefix] = item

    def keypaths(self) -> List[str]:
        """
        returns a list of keypaths (i.e. "a.b.c.d") to all values in the nested dict
        """
        # return NDict._keypaths_static(self._stored, None)
        return self._stored.keys()

    # @staticmethod
    # def _keypaths_static(item: Union[dict, Any], prefix: str) -> List[str]:
    #     if isinstance(item, MutableMapping):
    #         keys = []
    #         for key, value in item.items():
    #             if prefix is None:
    #                 cur_prefix = key
    #             else:
    #                 cur_prefix = f"{prefix}.{key}"
    #             keys += NDict._keypaths_static(value, cur_prefix)
    #         return keys
    #     else:
    #         return [prefix]

    def keys(self) -> dict_keys:
        """
        returns the top-level keys of the dictionary
        """
        return self._stored.keys()

    def values(self) -> dict_items:
        return self._stored.values()

    def items(self) -> dict_items:
        return self._stored.items()

    def merge(self, other: dict) -> NDict:
        """
        inplace merge between self and other.
        """
        other_flat = NDict(other).flatten()
        for k, v in other_flat.items():
            self[k] = v

        return

    def __getitem__(self, key: str) -> Any:
        """
        traverses the nested dict by the path extracted from splitting the key on '.', if key not found,
        optionally shows the possible closest options
        :param key: dot delimited keypath into the nested dict


        # OPT:
            if the key exists, return it's value.
            if it doesn't, check if it is a prefix of other keys.
                if it is, return them as dict.
                if not, raise an error
        """
        # nested_key = key.split(".")
        # if not nested_key[0] in self._stored:
        #     raise NestedKeyError(key, self)

        # value = self._stored
        # for sub_key in nested_key:
        #     if isinstance(value, MutableMapping) and sub_key in value:
        #         value = value.get(sub_key)
        #     else:
        #         raise NestedKeyError(key, self)

        # return value

        if key in self:
            return self._stored[key]

        if self.is_prefix(key):  # TODO can be more optimized. we pass here once and in the "get_sub_dict" once again
            # collect "sub-dict"
            return self.get_sub_dict(key)

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
            False
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
        res = dict()  # TODO maybe return it as NDict ?
        key = key + "."
        for kk in self.keypaths():
            if kk.startswith(key):
                sub_key = kk.replace(key, "", 1)
                res[sub_key] = self[kk]

        return NDict(res)

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

        # nested_key = key.split(".")
        # element = self._stored
        # for key in nested_key[:-1]:
        #     if key not in element:
        #         element[key] = {}
        #     element = element[key]

        # # set the value
        # element[nested_key[-1]] = value

        # store it "flat" (d["a.b.c"] = v, will have only one key)
        self._stored[key] = value

    def __delitem__(self, key: str) -> None:  # OPT
        # nested_key = key.split(".")
        # steps = len(nested_key)
        # value = self._stored
        # for step_idx, sep_key in enumerate(nested_key):
        #     if step_idx < steps - 1:
        #         value = value[sep_key]
        #     else:  # last step
        #         del value[sep_key]
        if key in self:
            del self._stored[key]
            return  # TODO maybe delete ??

        # delete entire branch
        if self.is_prefix(key):
            for kk in list(self.keypaths()):
                if kk.startswith(f"{key}."):
                    del self[kk]

    def get_closest_key(self, key: str) -> str:
        """
        For a given keypath, returns the longest valid keypath in the current nested dict
        :param key: a full keypath with dot delimiter
        """
        partial_key = []
        partial_ndict = self._stored
        parts = key.split(".")
        for k in parts:
            if isinstance(partial_ndict, MutableMapping) and k in partial_ndict:
                partial_key.append(k)
                partial_ndict = partial_ndict[k]
            else:
                break
        return ".".join(partial_key)

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

    def __contains__(self, o: str) -> bool:  # OPT
        return o in self._stored

    def get(self, key: str, default_value: Any = None) -> Any:
        if key not in self:
            return default_value
        return self[key]

    def get_multi(self, keys: Optional[List[str]] = None) -> NDict:
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
