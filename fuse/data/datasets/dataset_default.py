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

import copy
from collections import OrderedDict
from collections.abc import Hashable, Sequence
from operator import itemgetter
from typing import Any, Dict, List
from warnings import warn

import numpy as np

from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.datasets.dataset_base import DatasetBase
from fuse.data.ops.ops_common import OpCollectMarker
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.utils.sample import (
    create_initial_sample,
    get_sample_id,
    get_specific_sample_from_potentially_morphed,
)
from fuse.utils.multiprocessing.run_multiprocessed import (
    get_from_global_storage,
    run_multiprocessed,
)
from fuse.utils.ndict import NDict


class DatasetDefault(DatasetBase):
    def __init__(
        self,
        sample_ids: int | Sequence[Hashable] | None,
        static_pipeline: PipelineDefault | None = None,
        dynamic_pipeline: PipelineDefault | None = None,
        cacher: SamplesCacher | None = None,
        allow_uncached_sample_morphing: bool = False,
    ):
        """
        :param sample_ids: list of sample_ids included in dataset. Or:
                - An integer that describes only the size of the dataset. This is useful in massive datasets
                    (for example 100M samples). In such case, multiple functionalities will not be supported, mainly -
                    cacher, allow_uncached_sample_morphing and get_all_sample_ids
                - None. In this case, the dataset will not deal with sample ids. it is the user's responsibility to handle
                    iterations w.r.t the length of the dataset, as well as the index passed to __getitem__
                    this is useful for massive datasets, but when the sample ids are not expected to be running integets from 0 to a given length.

        :param static_pipeline:   static_pipeline, the output of this pipeline will be automatically cached.
        :param dynamic_pipeline:  dynamic_pipeline. applied sequentially after the static_pipeline, but not automatically cached.
                                changing it will NOT trigger recaching of the static_pipeline part.
        :param cacher: optional SamplesCacher instance which will be used for caching samples to speed up samples loading
        :param allow_uncached_sample_morphing:  when enabled, allows an Op, to return None, or to return multiple samples (in a list)

        """
        super().__init__()

        # store arguments
        self._cacher = cacher
        if isinstance(sample_ids, (int, np.integer)):
            if allow_uncached_sample_morphing:
                raise Exception(
                    "allow_uncached_sample_morphing is not allowed when providing sample_ids=an integer value"
                )
            if cacher is not None:
                raise Exception(
                    "providing a cacher is not allowed when providing sample_ids=an integer value"
                )
            self._sample_ids_mode = "running_int"
        elif sample_ids is None:
            self._sample_ids_mode = "external"
        else:
            self._sample_ids_mode = "explicit"

        # self._orig_sample_ids = sample_ids
        self._allow_uncached_sample_morphing = allow_uncached_sample_morphing

        # verify unique names for dynamic pipelines
        if dynamic_pipeline is not None and static_pipeline is not None:
            if static_pipeline.get_name() == dynamic_pipeline.get_name():
                raise Exception(
                    f"Detected identical name for static pipeline and dynamic pipeline ({static_pipeline.get_name(static_pipeline.get_name())}).\nThis is not allowed, please initiate the pipelines with different names."
                )

        if static_pipeline is None:
            static_pipeline = PipelineDefault(
                "dummy_static_pipeline", ops_and_kwargs=[]
            )
        if dynamic_pipeline is None:
            dynamic_pipeline = PipelineDefault(
                "dummy_dynamic_pipeline", ops_and_kwargs=[]
            )

        if dynamic_pipeline is not None:
            assert isinstance(
                dynamic_pipeline, PipelineDefault
            ), f"dynamic_pipeline may be None or a PipelineDefault instance. Instead got {type(dynamic_pipeline)}"

        if static_pipeline is not None:
            assert isinstance(
                static_pipeline, PipelineDefault
            ), f"static_pipeline may be None or a PipelineDefault instance. Instead got {type(static_pipeline)}"

        if self._allow_uncached_sample_morphing:
            warn(
                "allow_uncached_sample_morphing is enabled! It is a significantly slower mode and should be used ONLY for debugging"
            )

        self._static_pipeline = static_pipeline
        self._dynamic_pipeline = dynamic_pipeline
        self._orig_sample_ids = copy.deepcopy(sample_ids)

        self._created = False

    @property
    def static_pipeline(self) -> PipelineDefault | None:
        return self._static_pipeline

    @property
    def dynamic_pipeline(self) -> PipelineDefault | None:
        return self._dynamic_pipeline

    def create(self, num_workers: int = 0, mp_context: str | None = None) -> None:
        """
        Create the data set, including caching
        :param num_workers: number of workers. used only when caching is disabled and allow_uncached_sample_morphing is enabled
            set num_workers=0 to disable multiprocessing (more convenient for debugging)
            Setting num_workers for caching is done in cacher constructor.
        :param mp_context: "fork", "spawn", "thread" or None for multiprocessing default
        :return: None
        """
        self._output_sample_ids_info = None
        if self._cacher is not None:
            self._output_sample_ids_info = self._cacher.cache_samples(
                self._orig_sample_ids
            )
        elif self._allow_uncached_sample_morphing:
            _output_sample_ids_info_list = run_multiprocessed(
                DatasetDefault._process_orig_sample_id,
                [(sid, self._static_pipeline, False) for sid in self._orig_sample_ids],
                workers=num_workers,
                mp_context=mp_context,
                desc="dataset_default.sample_morphing",
            )

            self._output_sample_ids_info = OrderedDict()
            self._final_sid_to_orig_sid = {}
            for sample_in_out_info in _output_sample_ids_info_list:
                orig_sid, out_sids = sample_in_out_info[0], sample_in_out_info[1]
                self._output_sample_ids_info[orig_sid] = out_sids
                if out_sids is not None:
                    assert isinstance(out_sids, list)
                    for final_sid in out_sids:
                        self._final_sid_to_orig_sid[final_sid] = orig_sid

        if self._output_sample_ids_info is not None:  # sample morphing is allowed
            self._final_sample_ids = []
            for orig_sid, out_sids in self._output_sample_ids_info.items():
                if out_sids is None:
                    continue
                self._final_sample_ids.extend(out_sids)
        else:
            self._final_sample_ids = self._orig_sample_ids

        self._orig_sample_ids = (
            None  # should not be use after create. use self._final_sample_ids instead
        )
        self._created = True

    def get_all_sample_ids(self) -> List[Any]:
        if not self._created:
            raise Exception("you must first call create()")

        if self._sample_ids_mode != "explicit":
            raise Exception(
                "get_all_sample_ids is not supported when constructed with non explicit sample_ids"
            )

        return copy.deepcopy(self._final_sample_ids)

    def set_final_sample_ids(self, final_sample_ids: List[Any]) -> None:
        self._final_sample_ids = final_sample_ids

    def add_sample_ids_and_recreate(
        self, added_sample_ids: List[Any], **kwargs: Dict
    ) -> None:
        self._orig_sample_ids = self._final_sample_ids + added_sample_ids
        self._created = False
        self.create(**kwargs)

    def __getitem__(self, item: int | Hashable) -> NDict:
        """
        Get sample, read from cache if possible
        :param item: either int representing sample index or sample_id
        :return: sample_dict
        """
        return self.getitem(item)

    def getitem(
        self,
        item: int | Hashable,
        collect_marker_name: str | None = None,
        keys: Sequence[str] | None = None,
    ) -> NDict:
        """
        Get sample, read from cache if possible
        :param item: either int representing sample index or sample_id
        :param collect_marker_name: Optional, specify name of collect marker op to optimize the running time
        :param keys: Optional, return just the specified keys or everything available if set to None
        :return: sample_dict
        """
        if not self._created:
            raise Exception("you must first call create()")

        # get sample id
        if self._sample_ids_mode != "explicit":
            sample_id = item
            if self._sample_ids_mode == "running_int":  # allow using non int sample_ids
                if sample_id >= self._final_sample_ids:
                    raise IndexError(
                        f"Expecting {sample_id} to be smaller than {self._final_sample_ids}"
                    )

        elif not isinstance(item, (int, np.integer)):
            sample_id = item
        else:
            sample_id = self._final_sample_ids[item]

        # get collect marker info
        collect_marker_info = self._get_collect_marker_info(collect_marker_name)

        # read sample
        if self._cacher is not None:
            sample = self._cacher.load_sample(
                sample_id, collect_marker_info["static_keys_deps"]
            )

        if self._cacher is None:
            if not self._allow_uncached_sample_morphing:
                sample = create_initial_sample(sample_id)
                sample = self._static_pipeline(sample)
                if not isinstance(sample, dict):
                    raise Exception(
                        f'By default when caching is disabled sample morphing is not allowed, and the output of the static pipeline is expected to be a dict. Instead got {type(sample)}. You can use "allow_uncached_sample_morphing=True" to allow this, but be aware it is slow and should be used only for debugging'
                    )
            else:
                orig_sid = self._final_sid_to_orig_sid[sample_id]
                sample = create_initial_sample(orig_sid)
                sample = self._static_pipeline(sample)

                assert sample is not None
                sample = get_specific_sample_from_potentially_morphed(sample, sample_id)

        sample = self._dynamic_pipeline(
            sample, until_op_id=collect_marker_info["op_id"]
        )

        if not isinstance(sample, dict):
            raise Exception(
                f"The final output of dataset static (+optional dynamic) pipelines is expected to be a dict. Instead got {type(sample)}"
            )

        # get just required keys
        if keys is not None:
            sample = sample.get_multi(keys)

        return sample

    def _get_multi_multiprocess_func(self, args: Any) -> Any:
        sid, kwargs = args
        return self.getitem(sid, **kwargs)

    @staticmethod
    def _getitem_multiprocess(item: Hashable | int | np.integer) -> Any:
        """
        Getitem method used to optimize the running time in a multiprocess mode
        """
        dataset = get_from_global_storage("dataset_default_get_multi_dataset")
        kwargs = get_from_global_storage("dataset_default_get_multi_kwargs")
        return dataset.getitem(item, **kwargs)

    def get_multi(
        self,
        items: Sequence[int | Hashable] | None = None,
        workers: int = 10,
        verbose: int = 1,
        mp_context: str | None = None,
        desc: str = "dataset_default.get_multi",
        **kwargs: Any,
    ) -> List[Dict]:
        """
        See super class
        :param workers: number of processes to read the data. set to 0 to not use multi processing (useful when debugging).
        :param mp_context: "fork", "spawn", "thread" or None for multiprocessing default
        """
        if items is None:
            sample_ids = list(range(len(self)))
        else:
            sample_ids = items

        for_global_storage = {
            "dataset_default_get_multi_dataset": self,
            "dataset_default_get_multi_kwargs": kwargs,
        }

        list_sample_dict = run_multiprocessed(
            worker_func=self._getitem_multiprocess,
            copy_to_global_storage=for_global_storage,
            args_list=sample_ids,
            workers=workers,
            verbose=verbose,
            mp_context=mp_context,
            desc=desc,
        )
        return list_sample_dict

    def __len__(self) -> int:
        if not self._created:
            raise Exception("you must first call create()")

        if self._sample_ids_mode == "running_int":
            return self._final_sample_ids
        elif self._sample_ids_mode == "external":
            raise Exception(
                "__len__ is not defined where explicit sample_ids or an interer len are not provided."
            )

        return len(self._final_sample_ids)

    # internal methods

    @staticmethod
    def _process_orig_sample_id(args: Any) -> Any:
        """
        Process, without caching, single sample
        """
        orig_sample_id, pipeline, return_sample_dict = args
        sample = create_initial_sample(orig_sample_id)

        sample = pipeline(sample)

        output_sample_ids = None

        if sample is not None:
            output_sample_ids = []
            if not isinstance(sample, list):
                sample = [sample]
            for curr_sample in sample:
                output_sample_ids.append(get_sample_id(curr_sample))

        if not return_sample_dict:
            return orig_sample_id, output_sample_ids

        return orig_sample_id, output_sample_ids, sample

    def _get_collect_marker_info(self, collect_marker_name: str) -> dict:
        """
        Find the required collect marker (OpCollectMarker in the dynamic pipeline).
        See OpCollectMarker for more details
        :param collect_marker_name: name to identify the required collect marker
        :return: a dictionary with the required info - including: name, op_id and static_keys_deps.
        if collect_marker_name is None will return default instruct to run the entire dynamic pipeline
        """
        # default values for case collect marker info is not used
        if collect_marker_name is None:
            return {"name": None, "op_id": None, "static_keys_deps": None}

        # find the required collect markers and extract the info
        collect_marker_info = None
        for (op, _), op_id in reversed(
            zip(self._dynamic_pipeline.ops_and_kwargs, self._dynamic_pipeline._op_ids)
        ):
            if isinstance(op, OpCollectMarker):
                collect_marker_info_cur = op.get_info()
                if collect_marker_info_cur["name"] == collect_marker_name:
                    if collect_marker_info is None:
                        collect_marker_info = collect_marker_info_cur
                        collect_marker_info["op_id"] = op_id
                        # continue to make sure this is the only one
                    else:
                        # throw an error if found more than one collect marker
                        raise Exception(
                            f"Error: two collect markers with name {collect_marker_info} found in dynamic pipeline"
                        )
        if collect_marker_info is None:
            raise Exception(
                f"Error: didn't find collect marker with name {collect_marker_info} in dynamic pipeline."
            )

        return collect_marker_info

    def summary(self) -> str:
        sum = ""
        sum += f"Type: {type(self).__name__}\n"
        sum += f"Num samples: {len(self._final_sample_ids)}\n"
        # TODO
        # sum += f"Cacher: {self._cacher.summary()}"
        # sum += f"Pipeline static: {self._static_pipeline.summary()}"
        # sum += f"Pipeline dynamic: {self._dynamic_pipeline.summary()}"

        return sum

    def subset(self, indices: Sequence[int]) -> None:
        """
        Create a subset of the dataset by a given indices (inplace).

        Example:
            For the dataset '[-2, 1, 5, 3, 8, 5, 6]' and the indices '[1, 2, 5]', the subset is [1, 5, 5]

        :param items: indices of the subset - if None, the subset is the whole set.

        """
        if indices is None:
            # Do nothing, the subset is the whole dataset
            return

        if not self._created:
            raise Exception("you must first call create()")

        # grab the specified data
        self._final_sample_ids = itemgetter(*indices)(self._final_sample_ids)
