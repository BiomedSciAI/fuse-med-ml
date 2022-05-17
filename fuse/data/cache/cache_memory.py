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
Cache to Memory
"""
from multiprocessing import Manager
from typing import Hashable, Any, List

from fuse.data.cache.cache_base import FuseCacheBase


class FuseCacheMemory(FuseCacheBase):
    """
    Cache to Memory
    """

    def __init__(self):
        super().__init__()

        self.reset()

    def __contains__(self, key: Hashable) -> bool:
        """
        See base class
        """
        return key in self._cache_dict

    def __getitem__(self, key: Hashable) -> Any:
        """
        See base class
        """
        return self._cache_dict.get(key, None)

    def __delitem__(self, key: Hashable) -> None:
        """
        See base class
        """
        if not self._cache_enable:
            raise Exception('First start caching using function start_caching()')

        item = self._cache_dict.pop(key, None)

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """
        See base class
        """
        if not self._cache_enable:
            raise Exception('First start caching using function start_caching()')

        self._cache_dict[key] = value

    def save(self) -> None:
        """
        Not saving, moving back to  simple data structures
        """
        self._cache_enable = False
        self._cache_dict = dict(self._cache_dict)

    def exist(self) -> bool:
        """
        See base class
        """
        return len(self._cache_dict) > 0

    def reset(self) -> None:
        """
        See base class
        """
        self._cache_dict = {}

    def get_all_keys(self, include_none: bool = False) -> List[Hashable]:
        """
        See base class
        """
        if include_none:
            return list(self._cache_dict.keys())
        else:
            return [key for key, value in self._cache_dict.items() if value is not None]

    def start_caching(self, manager: Manager) -> None:
        """
        Moving to multiprocessing data structures
        """
        self._cache_enable = True
        # if manager is  None assume that the it's not multiprocessing caching
        if manager is not None:
            self._cache_dict = manager.dict(self._cache_dict)
