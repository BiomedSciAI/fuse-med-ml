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

import copy
import types
from typing import Any, MappingView, Union, List
from _collections_abc import dict_items, dict_keys, dict_values

class NDict(dict):
    """N(ested)Dict - wraps a python dict, and allows to access nested elements via '.' separated key desc

    NOTE: assumes that all keys (including nested) are:
        1. strings
        2. do not contain '.' within a single key, as '.' is used as a special symbol for accesing deeper level nested dict.

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
    
    def __init__(self, d: Union[dict, tuple, types.GeneratorType, NDict, None]=None, verbouse: bool =False):        
        """
        :param d: the data with which to populate the nested dictionary, in case of NDict it acts as a copy constructor, 
            otherwise we just set all the keys and values using the setitem function
        :param verbouse: set to true if you want verbouse key not found exceptions
        """
        self._stored = dict()
        self.verbouse=verbouse
        if type(d) is tuple :
            d = dict(d)
        elif isinstance(d, types.GeneratorType):
            d = dict(d)

        if type(d) is dict:
            for k,d in d.items():
                self[k] = d
        elif type(d) is NDict:
            self._stored = copy.deepcopy(d._stored)
    
        
    def items(self):
        return self._stored.items()

    #NDict custom methods
    def to_dict(self) -> dict:
        '''
        converts to standard python dict
        :param copy: set to None (default) to get access to the internal stored dict
        '''
        return self._stored
    
    def clone(self, deepcopy: bool =True):
        '''
        does a deep or a shallow copy, shallow copy means only top level keys are copied and the values are only referenced
        in deep copy, all values are copied recursively
        :param deepcopy: if true, does deep copy, otherwise does shalow copy
        '''
        if not deepcopy:
            return NDict(copy.copy(self._stored))
        else:
            return NDict(copy.deepcopy(self._stored))
    

    def flatten(self) -> dict:
        '''
        flattens the dictionary
        :returns dict
        
        For example:

        nx = NDict({'a': {'b': 14, 'c': 12}, 'c': 100, 'z': {'foo': {'boo': 111}}})
        print(nx.flatten())                
        {'a.b': 14, 'a.c': 12, 'c': 100, 'z.foo.boo': 111}

        #you can use it to get a list of the flat keys:
        print(nx.flatten().keys()) 
        '''

        all_keys = {}
        for key in self._stored:
            if isinstance(self._stored[key], dict):
                all_sub_keys = NDict(self[key]).flatten()
                keys_to_add = {f'{key}.{sub_key}':all_sub_keys[sub_key] for sub_key in all_sub_keys}
                all_keys.update(keys_to_add)
            else:
                all_keys[key] = self._stored[key]
        
        return all_keys

    def keypaths(self) -> List[str]:
        """
        returns a list of keypaths (i.e. "a.b.c.d") to all values in the nested dict
        """
        return list(self.flatten().keys())


    def merge(self, other: dict) -> NDict:
        """
        returns a new NDict which is a merge between the current and the other NDict, common values are overridden 
        """
        return NDict.combine(self, other)

    @staticmethod
    def combine(base: dict, other: dict) -> dict:
        '''
        Combines two dicts (each can be NDict or dict), starts with self and adds/overrides from other
        '''
        base_flat = NDict(base).flatten()
        other_flat = NDict(other).flatten()
        base_flat.update(other_flat)
        return NDict(base_flat)        

    def __getitem__(self, key: str) -> Any:        
        """
        traverses the nested dict by the path extracted from spliting the key on '.', if key not found,
        optionally shows the possible closest options
        :param key: dot delimited keypath into the nested dict
        """
        nested_key = key.split('.')        
        if not nested_key[0] in self._stored:
            raise NestedKeyError(key, self)
        
        value = self._stored        
        for sub_key in nested_key:
            if not isinstance(value, dict):
                raise NestedKeyError(key, self)
            value = value.get(sub_key)
        
        if isinstance(value, dict):
            value = NDict(value)
            
        return value

    def __setitem__(self, key: str, value: Any):
        """
        go over the the dictionary according to the path, create the nodes that does not exist
        :param key: the keypath
        :param value: value to set
        """
        nested_key = key.split('.')
        element = self._stored
        for key in nested_key[:-1]:
            if key not in element:
                element[key] = {}
            element = element[key]

        # set the value
        element[nested_key[-1]] = value
        

    def __delitem__(self, key: str):
        nested_key = key.split('.')     
        steps = len(nested_key)
        value = self._stored
        for step_idx, sep_key in enumerate(nested_key):
            if step_idx < steps-1:
                value = value[sep_key]
            else: #last step
                del value[sep_key]


    def get_closest_key(self, key: str) -> str:
        """
        For a given keypath, returns the longest valid keypath in the current nested dict
        :param key: a full keypath with dot delimiter
        """
        partial_key = []
        partial_ndict = self._stored
        parts = key.split('.')
        for k in parts:
            if isinstance(partial_ndict, dict) and k in partial_ndict:
               partial_key.append(k)
               partial_ndict = partial_ndict[k] 
            else:
                break
        return '.'.join(partial_key)
    
    def __reduce__(self):
        return super().__reduce__()

    def __iter__(self):
        return iter(self._stored)
    
    def __len__(self):
        return len(self._stored)

    def __str__(self):
        return str(self._stored)

    def __repr__(self):
        return repr(self._stored)
    
    def __contains__(self, o: str) -> bool:
        return o == self.get_closest_key(o)


class NestedKeyError(KeyError):
    def __init__(self, key: str, d: NDict) -> None:
        partial_key = d.get_closest_key(key)
        if partial_key == '':
            partial_ndict = d
        else:
            partial_ndict = d[partial_key]
        
        if isinstance(partial_ndict, NDict):
            options = str([f"{partial_key}.{k}" for k in partial_ndict.flatten().keys()])
            error_str = f'key {key} does not exist\n. Possible keys are: {options}'
            super().__init__(error_str)
        else:
            error_str = f'key {key} does not exist\n. Closest key is: {partial_key}'
            super().__init__(error_str)