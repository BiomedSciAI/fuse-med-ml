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

from abc import abstractmethod
from collections.abc import Hashable, Sequence
from typing import Dict, List

from torch.utils.data.dataset import Dataset


class DatasetBase(Dataset):
    @abstractmethod
    def create(self, **kwargs: dict) -> None:
        """
        Make the dataset operational: might include data caching, reloading and more.
        """
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> str:
        """
        Get string including summary of the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def get_multi(
        self, items: Sequence[int | Hashable] | None = None, *args: list, **kwargs: dict
    ) -> List[Dict]:
        """
        Get multiple items, optionally just some of the keys
        :param items: specify the list of sequence to read or None for all
        """
        raise NotImplementedError

    @abstractmethod
    def subset(self, indices: Sequence[int]) -> None:
        """
        Subset of a dataset at specified indices - inplace
        :param indices: indices of the samples that will remain in the subset
        """
        raise NotImplementedError
