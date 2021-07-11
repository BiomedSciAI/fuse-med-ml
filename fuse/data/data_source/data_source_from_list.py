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

from typing import Sequence, Hashable

from fuse.data.data_source.data_source_base import FuseDataSourceBase


class FuseDataSourceFromList(FuseDataSourceBase):
    """
    Simple DataSource that can be initialized with a Python list (or other sequence).
    Does nothing but passing the list to Dataset.
    """

    def __init__(self, list_of_samples: Sequence[Hashable] = []) -> None:
        self.list_of_samples = list_of_samples

    def get_samples_description(self):
        return self.list_of_samples

    def summary(self) -> str:
        summary_str = ''
        summary_str += 'FuseDataSourceFromList - %d samples\n' % len(self.list_of_samples)
        return summary_str
