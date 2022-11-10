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

from typing import List, Optional, Union, Sequence
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.utils.sample import get_sample_id

from torch.utils.data import Dataset

from fuse.data.ops.op_base import OpBase
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.utils.ndict import NDict


# Dataset processor
class OpReadDataset(OpBase):
    """
    Op that extract data from pytorch dataset that returning sequence of values and adds those values to sample_dict
    """

    def __init__(self, dataset: Dataset, sample_keys: Sequence[str]):
        """
        :param dataset: the pytorch dataset to convert. The dataset[i] expected to return sequence of values or a single value
        :param sample_keys: sequence keys - naming each value returned by dataset[i]
        """
        # store input arguments
        super().__init__()
        self._sample_keys = sample_keys
        self._dataset = dataset

    def __call__(self, sample_dict: NDict) -> Union[None, dict, List[dict]]:
        """
        See super class
        """
        # extact dataset index
        name, dataset_index = get_sample_id(sample_dict)

        # extract values
        sample_values = self._dataset[dataset_index]
        if not isinstance(sample_values, Sequence):
            sample_values = [sample_values]
        assert len(self._sample_keys) == len(
            sample_values
        ), f"Error: expecting dataset[i] to return {len(self._sample_keys)} to match sample keys"

        # add values to sample_dict
        for key, elem in zip(self._sample_keys, sample_values):
            sample_dict[key] = elem
        return sample_dict


class DatasetWrapSeqToDict(DatasetDefault):
    """
    Fuse Dataset Wrapper
    wraps pytorch sequence dataset (pytorch dataset in which each sample, dataset[i] is a sequence of values).
    Each value extracted from pytorch sequence dataset will be added to sample_dict.
    Plus this dataset inherits all DatasetDefault features

    Example:
        torch_seq_dataset = torchvision.datasets.MNIST(path, download=True, train=True)
        # wrapping torch dataset
        dataset = DatasetWrapSeqToDict(name='train', dataset=torch_seq_dataset, sample_keys=('data.image', 'data.label'))
        train_dataset.create()

        # get sample
        sample = train_dataset[index] # sample is a dict with keys: 'data.sample_id', 'data.image' and 'data.label'
    """

    def __init__(
        self,
        name: str,
        dataset: Dataset,
        sample_keys: Union[Sequence[str], str],
        cache_dir: Optional[str] = None,
        sample_ids: Optional[Sequence] = None,
        **kwargs,
    ):
        """
        :param name: name of the data extracted from dataset, typically: 'train', 'validation;, 'test'
        :param dataset: the dataset to extract the data from
        :param sample_keys: sequence keys - naming each value returned by dataset[i]
        :param cache_dir: Optional - provied a path in case caching is required to help optimize the running time
        :param sample_ids: Optional - subset of the data's sample ids.
        :param kwargs: optional, additional arguments to provide to DatasetDefault
        """
        if sample_ids is None:
            sample_ids = [(name, i) for i in range(len(dataset))]

        static_pipeline = PipelineDefault(name="staticp", ops_and_kwargs=[(OpReadDataset(dataset, sample_keys), {})])
        if cache_dir is not None:
            cacher = SamplesCacher("dataset_test_cache", static_pipeline, cache_dir, restart_cache=True)
        else:
            cacher = None
        super().__init__(sample_ids=sample_ids, static_pipeline=static_pipeline, cacher=cacher, **kwargs)
