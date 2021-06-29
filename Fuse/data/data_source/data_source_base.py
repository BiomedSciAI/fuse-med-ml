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
Data source base
"""
from abc import ABC, abstractmethod


class FuseDataSourceBase(ABC):

    @abstractmethod
    def get_samples_description(self):
        """
        :return: list of samples description
        """
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> str:
        """
        String summary of the object
        """
        raise NotImplementedError
