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
Base class for caching
"""
from abc import ABC, abstractmethod
from multiprocessing import Manager
from typing import Hashable, Any, List


class FuseCacheBase(ABC):

    @abstractmethod
    def __contains__(self, key: Hashable) -> bool:
        """
        return true if key is already in cache
        :param key: any kind of hashable key
        :return: boolean. True if exist.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: Hashable) -> Any:
        """
        Get an item from cache. Will raise an error if key does not exist
        :param key: any kind of hashable key
        :return: the item
        """
        raise NotImplementedError

    @abstractmethod
    def __delitem__(self, key: Hashable) -> None:
        """
        Delete key. Will raise an error if key does not exist
        :param key: any kind of hashable key
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key: Hashable, value: Any) -> None:
        """
        Set key. Will override previous value if already exist.
        :param key: any kind of hashable key
        :param value: any kind of value to sture
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        """
        Save data to cache
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def exist(self) -> bool:
        """
        return True if cache exist and contains the samples
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset cache and delete all data
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_keys(self, include_none: bool = False) -> List[Hashable]:
        """
        Get all keys currently cached
        :param include_none: include or filter 'none samples' which represents no samples or bad samples
        :return: List of keys
        """
        raise NotImplementedError

    def start_caching(self, manager: Manager) -> None:
        """
        start caching - the caching will be done in save().
        :param manager: multiprocessing manager to create shared data structures
        :return: None
        """
        raise NotImplementedError
