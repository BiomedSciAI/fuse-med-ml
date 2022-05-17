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
Fuse Dataset Base
"""
import pickle
from abc import abstractmethod
from enum import Enum
from typing import Any, List, Optional

from torch.utils.data.dataset import Dataset


class FuseDatasetBase(Dataset):
    """
    Abstract base class for Fuse dataset.
    All subclasses should overwrite the following abstract methods inherited from  torch.utils.data.Dataset
    `__getitem__`, supporting fetching a data sample for a given key.
    `__len__`, which is expected to return the size of the dataset
    And the ones listed below
    """

    class SaveMode(Enum):
        # store just the info required for inference
        INFERENCE = 1,
        # store all the info
        TRAINING = 2

    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(self, **kwargs) -> None:
        """
        Used to enable the instance
        Typically will load caching, etc
        :param kwargs: different parameters per subclass
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, index: Optional[int], key: Optional[str], use_cache: bool = False) -> Any:
        """
        Get input, ground truth or metadata of a sample.

        :param index: the index of the item or None for all
        :param key: string representing the exact information required, use None for all.
        :param use_cache: if true, will try to reload the sample from caching mechanism in case exist.
        :return: the required info of a single sample of a list of samples
        """
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, samples: List[Any]) -> Any:
        """
        collate list of samples into batch
        :param samples: list of samples
        :return: batch
        """
        raise NotImplementedError

    # misc
    @abstractmethod
    def summary(self, statistic_keys: Optional[List[str]] = None) -> str:
        """
        String summary of the object
        :param statistic_keys: Optional. list of keys to output statistics about.
        """
        raise NotImplementedError

    # save and load datasets
    @abstractmethod
    def get_instance_to_save(self, mode: SaveMode) -> 'FuseDatasetBase':
        """
        Create lite instance version of dataset with just the info required to recreate it
        :param mode: see SaveMode for available modes
        :return: the instance to save
        """
        raise NotImplementedError

    @staticmethod
    def save(dataset: 'FuseDatasetBase', mode: SaveMode, filename: str) -> None:
        """
        Static method save dataset to the disc (see SaveMode for available modes)
        :param dataset: the dataset to save
        :param mode: required mode to save
        :param filename: file name to use
        :return: None
        """
        # get instance version to save
        dataset_to_save = dataset.get_instance_to_save(mode)

        # save this instance
        with open(filename, 'wb') as pickle_file:
            pickle.dump(dataset_to_save, pickle_file)

    @staticmethod
    def load(filename: str, **kwargs) -> 'FuseDatasetBase':
        """
        load dataset
        :param filename: path to saved dataset
        :param kwargs: arguments of create() function
        :return: the dataset object
        """
        # load saved instance
        with open(filename, 'rb') as pickle_file:
            dataset = pickle.load(pickle_file)

        # recreate dataset
        dataset.create(**kwargs)

        return dataset
