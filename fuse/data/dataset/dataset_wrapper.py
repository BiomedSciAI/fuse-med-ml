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

from typing import Union, Sequence, Dict, Tuple

from torch.utils.data import Dataset

from fuse.data.data_source.data_source_from_list import FuseDataSourceFromList
from fuse.data.dataset.dataset_default import FuseDatasetDefault
from fuse.data.processor.processor_base import FuseProcessorBase


# Dataset processor
class DatasetProcessor(FuseProcessorBase):
    """
    Processor that extract data from pytorch dataset and convert each sample to dictionary
    """

    def __init__(self, dataset: Dataset, mapping: Sequence[str]):
        """
        :param dataset: the pytorch dataset to convert
        :param mapping: dictionary key for each element returned by the pytorch dataset
        """
        # store input arguments
        self.mapping = mapping
        self.dataset = dataset

    def __call__(self, desc: Tuple[str, int], *args, **kwargs):
        index = desc[1]
        sample = self.dataset[index]
        sample = {self.mapping[i]: val for i, val in enumerate(sample)}

        return sample


class FuseDatasetWrapper(FuseDatasetDefault):
    """
    Fuse Dataset Wrapper
    wraps pytorch dataset.
    Each sample will be converted to dictionary according to mapping.
    And this dataset inherits all FuseDatasetDefault features
    """

    #### CONSTRUCTOR
    def __init__(self, name: str, dataset: Dataset, mapping: Union[Sequence, Dict[str, str]], **kwargs):
        """
        :param name: name of the data extracted from dataset, typically: 'train', 'validation;, 'test'
        :param dataset: the dataset to extract the data from
        :param mapping: including name for each returned object from dataset
        :param kwargs: optinal, additional argumentes to provide to FuseDatasetDefault
        """
        data_source = FuseDataSourceFromList([(name, i) for i in range(len(dataset))])
        processor = DatasetProcessor(dataset, mapping)
        super().__init__(data_source=data_source, input_processors=None, gt_processors=None,processors=processor,  **kwargs)
