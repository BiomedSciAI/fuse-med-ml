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
Dummy cache implementation, doing nothing
"""
from multiprocessing import Manager
from typing import Hashable, Any, List

from fuse.data.cache.cache_base import FuseCacheBase


class FuseCacheNull(FuseCacheBase):
    def __init__(self):
        super().__init__()

    def __contains__(self, key: Hashable) -> bool:
        """
        See base class
        """
        return False

    def __getitem__(self, key: Hashable) -> Any:
        """
        See base class
        """
        return None

    def __delitem__(self, key: Hashable) -> None:
        """
        See base clas
        """
        pass

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """
        See base class
        """
        pass

    def save(self) -> None:
        """
        See base class
        """
        pass

    def exist(self) -> bool:
        """
        See base class
        """
        return True

    def reset(self) -> None:
        """
        See base class
        """
        pass

    def get_all_keys(self, include_none: bool = False) -> List[Hashable]:
        """
        See base class
        """
        return []

    def start_caching(self, manager: Manager):
        """
        See base class
        """
        pass
