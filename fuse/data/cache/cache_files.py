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
Cache to file per sample
"""
import gzip
import logging
import os
import pickle
import traceback
from multiprocessing import Manager
from typing import Hashable, Any, List
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from fuse.data.cache.cache_base import FuseCacheBase
from fuse.utils.utils_atomic_file import FuseUtilsAtomicFileWriter
from fuse.utils.utils_file import FuseUtilsFile


class FuseCacheFiles(FuseCacheBase):
    def __init__(self, cache_file_dir: str, reset_cache: bool, single_file: bool=False):
        """
        :param cache_file_dir: path to cache dir
        :param reset_cache: reset previous cache if exist or continue
        """
        super().__init__()

        self._cache_file_dir = cache_file_dir

        # create dir if not already exist
        FuseUtilsFile.create_dir(cache_file_dir)

        # pointer to cache index
        self._cache_file_name = os.path.join(self._cache_file_dir, 'cache_index.pkl')
        self._cache_prop_file_name = os.path.join(self._cache_file_dir, 'cache_properties.pkl')

        # reset or load from disk
        if reset_cache or not os.path.exists(self._cache_file_name):
            self.reset()
            self.single_file = single_file
            # save initial properties
            with FuseUtilsAtomicFileWriter(filename=self._cache_prop_file_name) as cache_prop_file:
                pickle.dump({'single_file': self.single_file}, cache_prop_file)
        else:
            # get last modified time of the index
            self._cache_index_mtime = os.path.getmtime(self._cache_file_name)

            # load current cache
            try:
                with open(self._cache_file_name, 'rb') as cache_index_file:
                    self._cache_index = pickle.load(cache_index_file)
            except:
                # backward compatibility - used to be saved in gz format
                with gzip.open(self._cache_file_name, 'rb') as cache_index_file:
                    self._cache_index = pickle.load(cache_index_file)
            self._cache_list = list(self._cache_index.keys())

            # load mode for backward compatibility
            try:
                with open(self._cache_prop_file_name, 'rb') as cache_prop_file:
                    cache_prop = pickle.load(cache_prop_file)
                    self.single_file = cache_prop['single_file']
            except:
                self.single_file = False

    def __contains__(self, key: Hashable) -> bool:
        """
        See base class
        """
        return key in self._cache_index

    def __getitem__(self, key: Hashable) -> Any:
        """
        See base class
        """
        if self.single_file:
            return self._cache_index.get(key, None)

        value_file_name = self._cache_index.get(key, None)
        if value_file_name is None:
            return None
        value_file_name = os.path.join(self._cache_file_dir, value_file_name)

        # make sure file not exist
        if os.path.exists(value_file_name):
            # store the file
            with gzip.open(value_file_name, 'rb') as value_file:
                value = pickle.load(value_file)
        else:
            raise Exception(f'cache file {value_file_name} not found')

        return value

    def __delitem__(self, key: Hashable) -> None:
        """
        Not supported
        """
        raise NotImplementedError

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """
        See base class
        """
        if not self._cache_enable:
            raise Exception('First start caching using function start_caching()')

        self._cache_list.append(key)

        # if value is none, just update cache index
        if value is None:
            self._cache_index[key] = None
            return
        if self.single_file:
            self._cache_index[key] = value
        else:
            index = self._cache_list.index(key)
            value_file_name = str(index).zfill(10) + '.pkl.gz'
            value_abs_file_name = os.path.join(self._cache_file_dir, value_file_name)
            self._cache_index[key] = value_file_name

            # make sure file not exist
            if os.path.exists(value_abs_file_name):
                logging.getLogger('Fuse').warning(f'cache file {value_abs_file_name} unexpectedly exist, overriding it.')

            # store the file
            with FuseUtilsAtomicFileWriter(value_abs_file_name) as value_file:
                pickle.dump(value, value_file)

            # store the cache index - just for a case of crashing
            try:
                with FuseUtilsAtomicFileWriter(filename=self._cache_file_name) as cache_index_file:
                    pickle.dump(dict(self._cache_index), cache_index_file)
            except:
                # do not trow error- just print warning
                lgr = logging.getLogger('Fuse')
                track = traceback.format_exc()
                lgr.warning(track)

    def save(self) -> None:
        """
        Save cache index file
        """
        # disable caching
        self._cache_enable = False

        with FuseUtilsAtomicFileWriter(filename=self._cache_file_name) as cache_index_file:
            pickle.dump(dict(self._cache_index), cache_index_file)

        # move back to simple data structures
        self._cache_index = dict(self._cache_index)
        self._cache_list = list(self._cache_list)

    def exist(self) -> bool:
        """
        See base class
        """
        return bool(self._cache_index)

    def reset(self) -> None:
        """
        See base class
        """
        # make sure the dir content is empty
        FuseUtilsFile.remove_dir_content(self._cache_file_dir)

        # create empty data structures
        self._cache_enable = False
        self._cache_index = {}
        self._cache_list = []
        self._cache_index_mtime = -1

    def get_all_keys(self, include_none: bool = False) -> List[Hashable]:
        """
        See base class
        """
        if include_none:
            return list(self._cache_index.keys())
        else:
            return [key for key, value in self._cache_index.items() if value is not None]

    def start_caching(self, manager: Manager):
        """
        See base class
        """
        self._cache_enable = True
        # if manager is  None assume that the it's not multiprocessing caching
        if manager is not None:
            # create dictionary and adds it one by one to workaround multiprocessing limitation
            cache_index = manager.dict()
            for k, v in self._cache_index.items():
                cache_index[k] = v
            self._cache_index = cache_index
            self._cache_list = manager.list(self._cache_list)
