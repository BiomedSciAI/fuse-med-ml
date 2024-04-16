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
import difflib

import copy
import types
from numpy import ndarray
from torch import Tensor
from typing import (
    Any,
    Callable,
    Iterator,
    Optional,
    Sequence,
    Union,
    List,
    MutableMapping,
)


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
        nx = NDict()
        nx["a.d.zz] = 42
        assert nx['a']['d.zz'] == 42

    """

    def __init__(
        self,
        dict_like: Union[dict, tuple, types.GeneratorType, NDict, None] = None,
        already_flat: bool = False,
    ):
        """
        :param dict_like: the data with which to populate the nested dictionary, in case of NDict it acts as view constructor,
            otherwise we just set all the keys and values using the setitem function
        :param already_flat: optimization option. set to True only if you are sure that the input "dict_like" is a dict without nested dictionaries.
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

    def to_dict(self) -> dict:
        """
        converts to standard python dict
        """
        return self._stored

    def clone(self, deepcopy: bool = True) -> NDict:
        """
        does a deep or a shallow copy, shallow copy means only top level keys are copied and the values are only referenced.
        in deep copy, all values are copied recursively

        :param deepcopy: if true, does deep copy, otherwise does a shallow copy
        """
        if not deepcopy:
            return NDict(copy.copy(self._stored), already_flat=True)
        else:
            return NDict(copy.deepcopy(self._stored), already_flat=True)

    def flatten(self) -> NDict:
        """
        Legacy from previous implementation of NDict where we store the data with nested dictionaries.
        currently we store the data with one flat dictionary
        """
        return self

    def keypaths(self) -> List[str]:
        """
        :return: a list of keypaths (i.e. "a.b.c.d") to all values in the nested dict
        """
        return list(self._stored.keys())

    def keys(self) -> dict_keys:
        """
        returns keypaths (threat as a "flat" dictionary)
        """
        return self._stored.keys()

    def top_level_keys(self) -> List[str]:
        """
        return top-level keys.

        Example:
            >>> ndict = NDict()
            >>> ndict["a1.b.c"] = 42
            >>> ndict["a2.b.h"] = 23
            >>> ndict.top_level_keys()
            ['a1', 'a2']
        """
        top_level_keys = {key.split(".")[0] for key in self.keys()}
        return list(top_level_keys)

    def values(self) -> dict_items:
        return self._stored.values()

    def items(self) -> dict_items:
        return self._stored.items()

    def merge(self, other: MutableMapping) -> None:
        """
        inplace merge between self and other.
        """
        for k, v in other.items():
            self[k] = v
        return

    def __getitem__(self, key: str) -> Any:
        """
        "traverses" the nested dict by the path extracted from splitting the key on '.'.
        if key not found, shows the possible closest options.
        if a prefix is given (the key exists but it is not a leaf), returns the corresponding sub dictionary

        :param key: dot delimited keypath into the nested dict
        """

        try:
            # the key is a full key for a value
            return self._stored[key]

        except KeyError:
            # the key is a prefix for other value(s)
            sub_dict = self.get_sub_dict(key)

            if sub_dict is not None:
                return sub_dict
            else:
                raise NestedKeyError(key, self)

    def get_sub_dict(self, key: str) -> Union[NDict, None]:
        """
        returns a copy of the "sub-dict" give a sub-key.
        if a sub-dict wasn't found, returns None

        Example:
            >>> ndict = NDict()
            >>> ndict["a.b.c"] = "x"
            >>> ndict["a.b.d"] = "y"
            >>> ndict["a.b.e"] = "z"
            >>> ndict.get_sub_dict("a.b")
            {'c': 'x', 'd': 'y', 'e': 'z'}
        """
        res = NDict()
        prefix_key = key + "."
        suffix_key = None
        for kk in self.keys():
            if kk.startswith(prefix_key):
                suffix_key = kk[len(prefix_key) :]
                res[suffix_key] = self[kk]

        if suffix_key is None:
            return None

        return res

    def __setitem__(self, key: str, value: Any) -> None:
        """
        sets item in the dictionary.
        if the item is a dict like object, it will parse the values using the delimiter "."

        Example:
            ndict = NDict()
            ndict["a"] = {"b" : {"c" : 42}}

            assert ndict["a.b.c"] == 42

        :param key: the keypath
        :param value: value to set
        """
        # if value is dictionary add to self key by key to avoid from keys with delimiter "."
        if isinstance(value, MutableMapping):
            for sub_key in value:
                self[f"{key}.{sub_key}"] = value[sub_key]
            return
        self._stored[key] = value

    def __delitem__(self, key: str) -> None:
        """
        deletes self[key] item(s).
        if key is a prefix to a sub-dict, deletes the entire sub-dict.
        if key is a key for a value AND a sub-dict, deletes BOTH
        """
        deleted = False

        # delete specific (key, value)
        if key in self._stored:
            del self._stored[key]
            deleted = True

        # delete entire branch
        for kk in list(self.keys()):
            if kk.startswith(f"{key}."):
                del self[kk]
                deleted = True

        if not deleted:
            raise NestedKeyError(key, self)

    def get_closest_keys(self, key: str, n: int = 1) -> List[str]:
        """
        For a given keypath, returns the closest key(s) in the current nested dict.
        if key exists in the dictionary
        :param key: string as a key
        :param n: the amount closest key(s) to return
        """
        if key in self._stored:
            return [key]

        total_sub_keys = []

        for kk in self.keys():
            splitted_key = kk.split(".")
            for i in range(len(splitted_key)):
                total_sub_keys.append(".".join(splitted_key[: i + 1]))

        closest_keys = difflib.get_close_matches(key, total_sub_keys, n=n, cutoff=0)
        return closest_keys

    def pop(self, key: str) -> Any:
        """
        returns the value self[key] and removes the key from the dict.

        :param key: the key to return and remove
        """
        res = self[key]
        del self[key]
        return res

    def indices(self, indices: ndarray) -> dict:
        """
        Extract the specified indices from each element in the dictionary (if possible)

        :param nested_dict: input dict
        :param indices: indices to extract. Either list of indices or boolean numpy array
        :return: NDict with only the required indices
        """
        new_dict = {}
        all_keys = self.keys()
        for key in all_keys:
            try:
                value = self[key]
                if isinstance(value, (ndarray, Tensor)):
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
        all_keys = self.keys()
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

    def __contains__(self, key: str) -> bool:
        if key in self._stored:
            return True

        key_dot = key + "."
        # iterate over all keys and looking for a match
        for kk in self.keys():
            if kk.startswith(key_dot):
                return True

        # a match wasn't found
        return False

    def get(self, key: str, default_value: Any = None) -> Any:
        if key not in self:
            return default_value
        return self[key]

    def get_multi(self, keys: Optional[List[str]] = None) -> NDict:
        """
        returns a subset of the dict with the specified keys

        :param keys: keys to keep in the returned ndict
        """
        if keys is None:
            keys = self.keys()  # take all keys

        ans = NDict()
        for k in keys:
            curr = self[k]
            ans[k] = curr

        return ans

    def unflatten(self) -> dict:
        """
        returns a nested dictionaries

        Example:

        >>> ndict = NDict({"a.b.c":42, "a.b.d": 23})
        >>> ndict.unflatten()
        {'a': {'b': {'d': 23, 'c': 42}}}
        """

        return NDict._unflatten_static(data_dict=self)

    @staticmethod
    def _unflatten_static(data_dict: NDict) -> dict:
        res = dict()
        for top_key in data_dict.top_level_keys():
            value = data_dict[top_key]
            if isinstance(value, NDict):
                res[top_key] = NDict._unflatten_static(value)
            else:
                res[top_key] = value

        return res

    def update(self, dict_like: MutableMapping) -> None:
        self.merge(dict_like)

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
        print(self.get_tree(print_values=print_values))

    def get_tree(self, print_values: bool = False) -> str:
        """
        returns a string of the NDict object in a tree-like structure. See 'print_tree'.
        """
        unflatten_dict = self.unflatten()
        tree = self._get_tree_str_static(unflatten_dict, print_values=print_values)
        return tree[:-1] if len(tree) > 0 else tree

    @staticmethod
    def _get_tree_str_static(
        data_dict: dict, level: int = 0, print_values: bool = False
    ) -> None:
        """
        static-method to print the inner structure of a dict in a tree-like structure.

        :param level: current recursive level inside the ndict
        :param print_values: set to True in order to also print ndict's stored values
        """
        res = ""
        keys = data_dict.keys()
        level += 1
        for key in keys:
            if isinstance(data_dict[key], dict):
                res += " ".join(["---" * level, key + "\n"])
                res += NDict._get_tree_str_static(
                    data_dict[key], level, print_values=print_values
                )
            else:
                if print_values:
                    res += " ".join(
                        ["---" * level, key, "->", str(data_dict[key]) + "\n"]
                    )
                else:
                    res += " ".join(["---" * level, key + "\n"])

        return res

    def describe(self) -> None:
        for k in self.keys():
            print(f"{k}")
            val = self[k]
            print(f"\ttype={type(val)}")
            if hasattr(val, "shape"):
                print(f"\tshape={val.shape}")


class NestedKeyError(KeyError):
    def __init__(self, key: str, d: NDict) -> None:
        closest_keys = d.get_closest_keys(key, n=3)
        if len(closest_keys) == 0:
            error_str = f"Error: key {key} does not exist\n. All keys: {d.keys()}"
        else:
            error_str = f"Error: key {key} does not exist\n. Top {len(closest_keys)} Closest key are: {closest_keys}. All keys: {d.keys()}"
        print(error_str)
        super().__init__(error_str)
