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

import logging
import os
from multiprocessing import Manager
from multiprocessing.pool import Pool, ThreadPool
from typing import Any, Dict, Optional, Hashable, List, Union, Tuple, Callable

import numpy as np
import torch
from pandas import DataFrame
from torch import Tensor
from tqdm import tqdm, trange

from fuse.data.augmentor.augmentor_base import FuseAugmentorBase
from fuse.data.cache.cache_base import FuseCacheBase
from fuse.data.cache.cache_files import FuseCacheFiles
from fuse.data.cache.cache_memory import FuseCacheMemory
from fuse.data.cache.cache_null import FuseCacheNull
from fuse.data.data_source.data_source_base import FuseDataSourceBase
from fuse.data.dataset.dataset_base import FuseDatasetBase
from fuse.data.processor.processor_base import FuseProcessorBase
from fuse.data.visualizer.visualizer_base import FuseVisualizerBase
from fuse.utils.utils_debug import FuseUtilsDebug
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
from fuse.utils.utils_logger import log_object_input_state
from fuse.utils.utils_misc import get_pretty_dataframe, FuseUtilsMisc


class FuseDatasetGenerator(FuseDatasetBase):
    """
    Fuse Dataset Generator
    Used when it's more convient to generate sevral samples at once
    """

    #### CONSTRUCTOR
    def __init__(self, data_source: FuseDataSourceBase, processor: FuseProcessorBase,
                 cache_dest: Optional[Union[str, int]] = None, augmentor: Optional[FuseAugmentorBase] = None,
                 visualizer: Optional[FuseVisualizerBase] = None, post_processing_func=None,
                 statistic_keys: Optional[List[str]] = None,
                 filter_keys: Optional[List[str]] = None):
        """
        :param data_source: objects provides the list of object description
        :param processor: data generator
        :param cache_dest: Optional, path to save caching
        :param augmentor: Optional, object that perform the augmentation
        :param visualizer: Optional, object that visualize the data
        :param post_processing_func: callback that allows to dynamically modify the data.
               Called as last step (after augmentation)
        :param statistic_keys: Optional. list of statistic keys to output in default self.summary() implementation
        :param filter_keys: Optional. list of keys to remove from the sample dictionary when getting an item
        """
        # log object input state
        log_object_input_state(self, locals())

        super().__init__()

        # store input params
        self.cache_dest = cache_dest
        self.augmentor = augmentor
        self.visualizer = visualizer
        self.processor = processor
        self.data_source = data_source
        self.post_processing_func = post_processing_func
        self.statistic_keys = statistic_keys or []
        self.filter_keys = filter_keys or []
        # initial values
        # map sample running index to sample description (mush be hashable)
        self.subsets_description = []

        # create default cache for now - the cache will be created and loaded in create()
        self.cache: FuseCacheBase = FuseCacheMemory()
        # create dummy cache
        # self.cache_fields is used to store specific fields of the sample -
        # used to optimize the running time of dataset.get(key=<key name>, use_cache=True)
        self.cache_fields: FuseCacheBase = FuseCacheNull()

        # debug modes - read configuration
        self.sample_stages_debug = FuseUtilsDebug().get_setting('dataset_sample_stages_info') != 'default'
        self.sample_user_debug = FuseUtilsDebug().get_setting('dataset_user') != 'default'

    def create(self, reset_cache: bool = False,
               num_workers: int = 16, worker_init_func: Callable = None, worker_init_args: Any = None,
               override_datasource: Optional[FuseDataSourceBase] = None, override_cache_dest: Optional[str] = None,
               pool_type: str = 'process') -> None:

        """
        Create the data set, including loading sample descriptions and caching
        :param reset_cache: if False and cache_all is True, will use load caching instead of re creating it.
        :param num_workers: number of workers used for caching
        :param worker_init_func: process initialization function (multi processing mode)
        :param worker_init_args: worker init function arguments
        :param override_datasource: might be used to change the data source
        :param override_cache_dest: might be user to change the cache destination
        :param pool_type: multiprocess pooling type, can be either 'thread' (for ThreadPool) or 'process' (for 'Pool', default).
        :return: None
        """
        # debug - override num workers
        override_num_workers = FuseUtilsDebug().get_setting('dataset_override_num_workers')
        if override_num_workers != 'default':
            num_workers = override_num_workers
            logging.getLogger('Fuse').info(f'Dataset - debug mode - override num workers to {override_num_workers}', {'color': 'red'})

        assert pool_type in ['thread', 'process'], f'Invalid pool_type: {pool_type}. Multiprocessing pooling type can be either "thread" or "process"'
        self.pool_type = pool_type

        # override data source if required
        if override_datasource is not None:
            self.data_source = override_datasource
        # override destination cache if required
        if override_cache_dest is not None:
            self.cache_dest = override_cache_dest
        # extract list of sample description
        self.subsets_description = self.data_source.get_samples_description()

        # debug - override number of samples
        dataset_override_num_samples = FuseUtilsDebug().get_setting('dataset_override_num_samples')
        if dataset_override_num_samples != 'default':
            self.subsets_description = self.subsets_description[:dataset_override_num_samples]
            logging.getLogger('Fuse').info(f'Dataset - debug mode - override num samples to {dataset_override_num_samples}', {'color': 'red'})

        # cache object
        if isinstance(self.cache_dest, str) and self.cache_dest == 'memory':
            self.cache: FuseCacheBase = FuseCacheMemory()
        elif isinstance(self.cache_dest, str):
            self.cache: FuseCacheBase = FuseCacheFiles(self.cache_dest, reset_cache)

        # cache samples if required
        if not isinstance(self.cache, FuseCacheNull):
            self.cache_all_samples(num_workers=num_workers, worker_init_func=worker_init_func, worker_init_args=worker_init_args)

            # update descriptors
            all_descriptors = self.subsets_description
            cached_descriptors = self.cache.get_all_keys()
            self.samples_description = sorted([desc for desc in cached_descriptors if desc[0] in all_descriptors])

        self.sample_descriptor_to_index = {v: k for k, v in enumerate(self.samples_description)}
    #### ITERATE AND GET DATA
    def __len__(self):
        return len(self.samples_description)

    def get(self, index: Optional[Union[int, Hashable]], key: Optional[str] = None, use_cache: bool = True) -> Any:
        """
        Get input, ground truth or metadata of a sample.

        :param index: the index of the item, if None will return all items.
                      If not an int or None, will assume that imdex is sample descriptor
        :param key: string representing the exact information required. If None, will return all sample
        :param use_cache: if true, will try to reload the sample from caching mechanism
        :return: the required info
        """
        if index is not None and not isinstance(index, int):
            # get sample giving sample descriptor
            # assume index is sample description
            index = self.samples_description.index(index)

        # if key not specified return the all sample
        if key is None:
            assert index != -1, 'get all samples is not supported when key = None'
            return self.getitem(index)

        assert use_cache == True, f'{type(self)} support only use_cache=True'

        if index is None:
            # return all samples
            values = []
            for index in trange(len(self)):
                # first look for the specific file inside the cache
                desc_field = (self.samples_description[index], key)
                if desc_field in self.cache_fields:
                    values.append(self.cache_fields[desc_field])
                else:
                    # if not found get the all sample and then extract the specified field
                    values.append(FuseUtilsHierarchicalDict.get(self.getitem(index, apply_augmentation=False), key))
            return values
        else:
            # return single sample
            # first look for the specific file inside the cache
            desc_field = (self.samples_description[index], key)
            if desc_field in self.cache_fields:
                return self.cache_fields[desc_field]
            else:
                # if not found get the all sample and then extract the specified field
                return FuseUtilsHierarchicalDict.get(self.getitem(index, apply_augmentation=False), key)

    def __getitem__(self, index: int) -> Any:
        """
        Get sample, read it from cache if possible, apply augmentation and post processing
        :param index: sample index
        :return: the required sample after augmentation
        """
        sample_stages_debug = self.sample_stages_debug
        return self.getitem(index, sample_stages_debug=sample_stages_debug)

    def getitem(self, index: int, apply_augmentation: bool = True, apply_post_processing: bool = True, sample_stages_debug: bool = False) -> Any:
        """
        Get sample, read it from cache if possible
        :param index: sample index
        :param apply_augmentation: if true, will apply augmentation
        :param apply_post_processing: If true, will apply post processing
        :param sample_stages_debug: True will log the sample dict after each stage
        :return: the required sample after augmentation
        """

        # load from cache
        sample_desc = self.samples_description[index]
        sample = self.cache[sample_desc]

        # filter some of the keys if required
        if self.filter_keys is not None:
            for key in self.filter_keys:
                try:
                    FuseUtilsHierarchicalDict.pop(sample, key)
                except KeyError:
                    pass

        # debug mode - print original sample before augmentation and before post processing
        if sample_stages_debug:
            lgr = logging.getLogger('Fuse')
            sample_str = FuseUtilsMisc.batch_dict_to_string(sample)
            lgr.info(f'Dataset - original sample:', {'color': 'green', 'attrs': 'bold'})
            lgr.info(f'{sample_str}', {'color': 'green'})
            # one time print
            self.sample_stages_debug = False

        # apply augmentation if enabled
        if self.augmentor is not None and apply_augmentation:
            sample = self.augmentor(sample)

            # debug mode - print sample after augmentation
            if sample_stages_debug:
                lgr = logging.getLogger('Fuse')
                sample_str = FuseUtilsMisc.batch_dict_to_string(sample)
                lgr.info(f'Dataset - augmented sample:', {'color': 'green', 'attrs': 'bold'})
                lgr.info(f'{sample_str}', {'color': 'green'})

        # apply post processing
        if self.post_processing_func is not None and apply_post_processing:
            self.post_processing_func(sample)

            # debug mode - print sample after post processing
            if sample_stages_debug:
                lgr = logging.getLogger('Fuse')
                sample_str = FuseUtilsMisc.batch_dict_to_string(sample)
                lgr.info(f'Dataset - post processed sample:', {'color': 'green', 'attrs': 'bold'})
                lgr.info(f'{sample_str}', {'color': 'green'})

        return sample

    #### BATCHING
    def collate_fn(self, samples: List[Dict], avoid_stack_keys: Tuple = tuple()) -> Dict:
        """
        collate list of samples into batch_dict
        :param samples: list of samples
        :param avoid_stack_keys: list of keys to just collect to a list and avoid stack operation
        :return: batch_dict
        """
        batch_dict = {}
        keys = FuseUtilsHierarchicalDict.get_all_keys(samples[0])
        for key in keys:
            try:
                collected_value = [FuseUtilsHierarchicalDict.get(sample, key) for sample in samples if sample is not None]
                if key in avoid_stack_keys:
                    FuseUtilsHierarchicalDict.set(batch_dict, key, collected_value)
                elif isinstance(collected_value[0], Tensor):
                    FuseUtilsHierarchicalDict.set(batch_dict, key, torch.stack(collected_value))
                elif isinstance(collected_value[0], np.ndarray):
                    FuseUtilsHierarchicalDict.set(batch_dict, key, np.stack(collected_value))
                else:
                    FuseUtilsHierarchicalDict.set(batch_dict, key, collected_value)
            except:
                logging.getLogger('Fuse').error(f'Failed to collect key {key}')
                raise

        return batch_dict

    #### CACHING
    def cache_all_samples(self, num_workers: int = 16, worker_init_func: Callable = None, worker_init_args: Any = None) -> None:
        """
        Cache all data
        :param num_workers: num of workers used to cache the samples
        :param worker_init_func: process initialization function (multi processing mode)
        :param worker_init_args: worker init function arguments
        :return: None
        """
        lgr = logging.getLogger('Fuse')

        # check if cache is required
        all_descriptors = set([(subset_desc, 0) for subset_desc in self.subsets_description])
        cached_descriptors = set(self.cache.get_all_keys(include_none=True))
        descriptors_to_cache = all_descriptors - cached_descriptors

        if len(descriptors_to_cache) != 0:
            # multi process cache
            lgr.info(f'FuseDatasetGenerator: caching {len(descriptors_to_cache)} out of {len(all_descriptors)}')
            with  Manager() as manager:
                # change cache mode - to caching (writing)
                self.cache.start_caching(manager)

                # multi process cache
                if num_workers > 0:
                    the_pool = ThreadPool if self.pool_type == 'thread' else Pool
                    pool = the_pool(processes=num_workers, initializer=worker_init_func, initargs=worker_init_args)
                    for _ in tqdm(pool.imap_unordered(func=self._cache_subset,
                                                      iterable=[(self.processor, subset_desc[0], self.cache) for subset_desc in descriptors_to_cache]),
                                  total=len(descriptors_to_cache), smoothing=0.1):
                        pass
                    pool.close()
                    pool.join()
                else:
                    for subset_desc in tqdm(descriptors_to_cache):
                        self._cache_subset((self.processor, subset_desc[0], self.cache))

                # save and move back to read mode
                self.cache.save()
                lgr.info('FuseDatasetGenerator: caching done')
        else:
            lgr.info('FuseDatasetGenerator: all samples are already cached')

    @staticmethod
    def _cache_subset(args: Tuple) -> None:
        """
        Store in cache single sample
        :param args: tuple of processor and subset descriptor
        :return: None
        """
        processor, subset_desc, cache = args
        samples = processor(subset_desc)
        if not isinstance(samples, List):
            samples = [samples]
        if samples:
            for sample_index, sample_data in enumerate(samples):

                assert isinstance(sample_data, dict), f'expecting sample_data to be dictionary, got {type(sample_data)}'
                sample_data = sample_data.copy()

                sample = {'data': sample_data}
                sample_data['descriptor'] = (subset_desc, sample_index)
                cache[sample_data['descriptor']] = sample
        else:
            # no samples extracted mark it as an invalid descriptor
            cache[(subset_desc, 0)] = None

    def cache_sample_fields(self, fields: List[str], reset_cache: bool = False, num_workers: int = 8, cache_dest: Optional[str] = None) -> None:
        """
        Cache specific fields (keys in batch_dict)
        Used to optimize the running time of of dataset.get(key=<key name>, use_cache=True)
        :param fields: list of keys in batch_dict
        :param reset_cache: If True will reset cache first
        :param num_workers: num workers used for caching
        :param cache_dest: path to cache dir
        :return: None
        """
        lgr = logging.getLogger('Fuse')

        # debug - override num workers
        override_num_workers = FuseUtilsDebug().get_setting('dataset_override_num_workers')
        if override_num_workers != 'default':
            num_workers = override_num_workers
            lgr.info(f'Dataset - debug mode - override num workers to {override_num_workers}', {'color': 'red'})

        if cache_dest is None:
            cache_dest = os.path.join(self.cache_dest, 'fields')

        # create cache field object upon request
        if isinstance(self.cache_fields, FuseCacheNull):
            # cache object
            if isinstance(cache_dest, str) and cache_dest == 'memory':
                self.cache_fields: FuseCacheBase = FuseCacheMemory()
            elif isinstance(cache_dest, str):
                self.cache_fields: FuseCacheBase = FuseCacheFiles(cache_dest, reset_cache, single_file=True)

        # get list of desc to cache
        desc_list = self.samples_description
        desc_field_list = set([(desc, field) for desc in desc_list for field in fields])
        cached_desc_field = set(self.cache_fields.get_all_keys(include_none=True))
        desc_field_to_cache = desc_field_list - cached_desc_field
        desc_to_cache = set([desc_field[0] for desc_field in desc_field_to_cache])

        # multi thread caching
        if len(desc_to_cache) != 0:
            lgr.info(f'FuseDatasetGenerator: samples fields - caching {len(desc_to_cache)} out of {len(desc_list)}')
            if num_workers > 0:
                with Manager() as manager:
                    self.cache_fields.start_caching(manager)
                    pool = Pool(processes=num_workers)
                    for _ in tqdm(pool.imap_unordered(func=self._cache_sample_fields,
                                                      iterable=[(desc, fields) for desc in desc_to_cache]),
                                  total=len(desc_to_cache), smoothing=0.1):
                        pass
                    pool.close()
                    pool.join()
                    self.cache_fields.save()
            else:
                self.cache_fields.start_caching(None)
                for desc in tqdm(desc_to_cache):
                    self._cache_sample_fields((desc, fields))
                self.cache_fields.save()
        else:
            lgr.info('FuseDatasetGenerator: all samples fields are already cached')

    def _cache_sample_fields(self, args):
        # decode args
        desc, fields = args
        index = self.samples_description.index(desc)
        sample = self.getitem(index, apply_augmentation=False)
        for field in fields:
            # create field desc and save it in cache
            desc_field = (desc, field)
            if desc_field not in self.cache_fields:
                value = FuseUtilsHierarchicalDict.get(sample, field)
                self.cache_fields[desc_field] = value

    #### Filtering
    def filter(self, key: str, values: List[Any]) -> None:
        """
        Filter sample if batch_dict[key] in values
        :param key: key in batch_dict
        :param values: list of values to filter
        :return: None
        """
        lgr = logging.getLogger('Fuse')
        lgr.info(f'DatasetGenerator: filtering key {key}, values {values}')
        new_samples_desc = []
        for index, desc in tqdm(enumerate(self.samples_description), total=len(self.samples_description)):
            value = self.get(index, key, use_cache=True)
            if value not in values:
                new_samples_desc.append(desc)

        self.samples_description = new_samples_desc



    #### VISUALISE
    def visualize(self, index: Optional[int] = None, descriptor: Optional[Hashable] = None, block: bool = True):
        """
        visualize sample
        :param index: sample index, only one of index/descriptor can be provided
        :param descriptor: descriptor of a sample , only one of index/descriptor can be provided
        :param block: set to False if the process should not be blocked until the plot will be closed
        :return: None
        """
        assert (index is not None) ^ (descriptor is not None), "visualize method must get one and one only of an index or a descriptor"
        lgr = logging.getLogger('Fuse')
        if descriptor is not None:
            index = self.sample_descriptor_to_index[descriptor]

        if self.visualizer is None:
            lgr.warning('Cannot visualize - visualizer was not provided')
            return

        batch_dict = self.getitem(index)

        self.visualizer.visualize(batch_dict, block)

    def visualize_augmentation(self, index: Optional[int] = None, descriptor: Optional[Hashable] = None, block: bool = True):
        """
        visualize augmentation of a sample
        :param index: sample index, only one of index/descriptor can be provided
        :param descriptor: descriptor of a sample, only one of index/descriptor can be provided
        :param block: set to False if the process should not be blocked until the plot will be closed
        :return: None
        """

        assert (index is not None) ^ (descriptor is not None), "visualize method must get one and one only of an index or a descriptor"

        lgr = logging.getLogger('Fuse')
        if descriptor is not None:
            index = self.sample_descriptor_to_index[descriptor]
        if self.visualizer is None:
            lgr.warning('Cannot visualize - visualizer was not provided')
            return
        batch_dict = self.getitem(index, apply_augmentation=False)
        batch_dict_aug = self.getitem(index)

        self.visualizer.visualize_aug(batch_dict, batch_dict_aug, block)

    # save and load dataset
    def get_instance_to_save(self, mode: FuseDatasetBase.SaveMode) -> FuseDatasetBase:
        """
        See base class
        """

        # prepare data to save
        if mode == FuseDatasetBase.SaveMode.INFERENCE:
            dataset = FuseDatasetGenerator(data_source=None,
                                           processor=self.processor,
                                           augmentor=self.augmentor,
                                           post_processing_func=self.post_processing_func
                                           )
        elif mode == FuseDatasetBase.SaveMode.TRAINING:
            dataset = FuseDatasetGenerator(data_source=self.data_source,
                                           processor=self.processor,
                                           augmentor=self.augmentor,
                                           post_processing_func=self.post_processing_func,
                                           visualizer=self.visualizer)
        else:
            raise Exception(f'Unexpected SaveMode {mode}')

        return dataset

    # misc
    def summary(self, statistic_keys: Optional[List[str]] = None) -> str:
        """
        Returns a data summary.
        Should be called after create()
        :param statistic_keys: Optional. list of keys to output statistics about.
                        When None (default), self.statistic_keys are output.
        :return: str
        """
        statistic_keys_to_use = statistic_keys if statistic_keys is not None else self.statistic_keys
        sum = \
            f'Class = {self.__class__}\n'
        sum += \
            f'Processor:\n' \
                f'-----------------\n' \
                f'{self.processor}\n'
        sum += \
            f'Cache destination:\n' \
                f'------------------\n' \
                f'{self.cache_dest}\n'
        sum += \
            f'Augmentor:\n' \
                f'----------\n' \
                f'{self.augmentor.summary() if self.augmentor is not None else None}\n'
        sum += \
            f'Sample keys:\n' \
                f'------------\n' \
                f'{FuseUtilsHierarchicalDict.get_all_keys(self.getitem(0)) if self.data_source is not None else None}\n'
        if len(statistic_keys_to_use) > 0:
            for key in statistic_keys_to_use:
                values = self.get(index=None, key=key, use_cache=True)
                # convert to int - maybe we will need to supporty additional types one day
                values = [int(value) for value in values]
                df = DataFrame(data=values,
                               columns=[key])
                stat_df = DataFrame()
                stat_df['Value'] = df[key].value_counts().index
                stat_df['Count'] = df[key].value_counts().values
                stat_df['Percent'] = df[key].value_counts(normalize=True).values * 100
                sum += \
                    f'\n{key} Statistics:\n' + \
                    f'{get_pretty_dataframe(stat_df)}'
        return sum
