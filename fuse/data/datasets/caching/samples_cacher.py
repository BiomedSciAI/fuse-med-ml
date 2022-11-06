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
from functools import partial
from typing import Hashable, List, Optional, Sequence, Union, Callable, Any, Tuple

from fuse.data.pipelines.pipeline_default import PipelineDefault
from collections import OrderedDict
from fuse.data.datasets.caching.object_caching_handlers import _object_requires_hdf5_recurse
from fuse.utils.ndict import NDict
import os
import psutil
from fuse.utils.file_io.file_io import load_hdf5, save_hdf5_safe, load_pickle, save_pickle_safe
from fuse.data import get_sample_id, create_initial_sample, get_specific_sample_from_potentially_morphed
import hashlib
from fuse.utils.file_io import delete_directory_tree
from glob import glob
from fuse.utils.multiprocessing.run_multiprocessed import run_multiprocessed, get_from_global_storage
from fuse.data.datasets.sample_caching_audit import SampleCachingAudit
from fuse.data.utils.sample import get_initial_sample_id, set_initial_sample_id
from warnings import warn


class SamplesCacher:
    def __init__(
        self,
        unique_name: str,
        pipeline: PipelineDefault,
        cache_dirs: Union[str, List[str]],
        custom_write_dir_callable: Optional[Callable] = None,
        custom_read_dirs_callable: Optional[Callable] = None,
        restart_cache: bool = False,
        workers: int = 0,
        verbose=1,
        use_pipeline_hash: Optional[bool] = True,
        **audit_kwargs: dict,
    ) -> None:
        """
        Supports caching samples, used by datasets implementations.
        :param unique_name: a unique name for this cache.
         cache dir will be [cache dir]/[unique_name]
        :param cache_dirs: a path in which the cache will be created,
         you may provide a list of paths, which will be tried in order, moving the next when available space is exausted.
        :param parameter:
        :param custom_write_dir_callable: optional callable with the signature foo(cache_dirs:List[str]) -> str
         which returns the write directory to use.
        :param custom_read_dirs_callable: optional callable with the signature foo() -> List[str]
         which returns a list of directories to attempt to read from. Attempts will be in the provided order.
        :param restart_cache: if set to True, will DELETE all of the content of the defined cache dirs.
        Should be used every time that any of the OPs participating in the "static cache" part changed in any way
        (for example, code change)
        :param workers: number of multiprocessing workers used when building the cache. Default value is 0 (no multiprocessing)
        :param use_pipeline_hash [Optional]: indicates whether to use a hash of given pipeline for naming its cache dir. Default=True
        :param **audit_kwargs: optional custom kwargs to pass to SampleCachingAudit instance.
            auditing cached samples (usually periodically) is very important, in order to avoid "stale" cached samples.
            To disable pass audit_first_sample=False, audit_rate=None,
            Note that it's not recommended to completely disable it, and at the very least you should use audit_first_sample=True, audit_rate=None
            which only tests the first loaded sample for staleness.
            To learn more read SampleCachingAudit doc
        """
        if not isinstance(cache_dirs, list):
            cache_dirs = [cache_dirs]
        self._cache_dirs = [os.path.join(x, unique_name) for x in cache_dirs]

        self._unique_name = unique_name

        if custom_write_dir_callable is None:
            self._write_dir_logic = _get_available_write_location
        else:
            self._write_dir_logic = custom_write_dir_callable

        if custom_read_dirs_callable is None:
            self._read_dirs_logic = partial(default_read_dirs_logic, cache_dirs=self._cache_dirs)
        else:
            self._read_dirs_logic = custom_read_dirs_callable

        self._pipeline = pipeline
        self._use_pipeline_hash = use_pipeline_hash

        self._pipeline_desc_text = str(pipeline)
        if use_pipeline_hash:
            self._pipeline_desc_hash = "hash_" + hashlib.md5(self._pipeline_desc_text.encode("utf-8")).hexdigest()
        else:
            self._pipeline_desc_hash = "hash_fixed"

        self._verbose = verbose

        if self._verbose > 0:
            print(f"pipeline description hash for [{unique_name}] is: {self._pipeline_desc_hash}")

        self._restart_cache = restart_cache
        if self._restart_cache:
            self.delete_cache()

        self._audit_kwargs = audit_kwargs
        self._audit = SampleCachingAudit(**self._audit_kwargs)

        self._workers = workers
        if self._workers < 2:
            warn(
                'Multi processing is not active in SamplesCacher. Seting "workers" to the number of your cores usually results in a significant speedup. Debugging, however, is easier with "workers=0".'
            )

        self._verify_no_other_pipelines_cache()

    def _verify_no_other_pipelines_cache(self) -> None:
        dirs_to_check = self._get_read_dirs() + [self._get_write_dir()]
        for d in dirs_to_check:
            search_pat = os.path.realpath(os.path.join(d, "..", "hash_*"))
            found_sub_dirs = glob(search_pat)
            for found_dir in found_sub_dirs:
                if not os.path.isdir(found_dir):
                    continue
                if os.path.basename(found_dir) != self._pipeline_desc_hash:
                    raise Exception(
                        f"Found samples cache for pipeline hash {os.path.basename(found_dir)} which is different from the current loaded pipeline hash {self._pipeline_desc_hash} !!\n"
                        "This is not allowed, you may only use a single pipeline per uniquely named cache.\n"
                        'You can use "restart_cache=True" to rebuild the cache or delete the different cache manually.\n'
                    )

    def delete_cache(self) -> None:
        """
        Will delete this specific named cache from all read and write dirs
        """
        dirs_to_delete = self._get_read_dirs() + [self._get_write_dir()]
        dirs_to_delete = list(set(dirs_to_delete))
        dirs_to_delete = [
            os.path.realpath(os.path.join(x, "..")) for x in dirs_to_delete
        ]  # one dir above the pipeline hash dir
        print('Due to "delete_cache" call, about to delete the following dirs:')

        for del_dir in dirs_to_delete:
            print(del_dir)
        print("---- list end ----")
        print("deleting ... ")
        for del_dir in dirs_to_delete:
            print(f"deleting {os.path.abspath(del_dir)} ...")
            delete_directory_tree(del_dir)

    def _get_write_dir(self):
        ans = self._write_dir_logic(self._cache_dirs)
        ans = os.path.join(ans, self._pipeline_desc_hash)
        return ans

    def _get_read_dirs(self):
        ans = self._read_dirs_logic()
        ans = [os.path.join(x, self._pipeline_desc_hash) for x in ans]
        return ans

    def cache_samples(self, orig_sample_ids: List[Any]) -> List[Tuple[str, Union[None, List[str]], str]]:
        """
        Go over all of orig_sample_ids, and cache resulting samples
        returns information that helps to map from original sample id to the resulting sample id
        (an op might return None, discarding a sample, or optional generate different one or more samples from an original single sample_id)
        #TODO: have a single doc location that explains this concept and can be pointed to from any related location
        """
        # TODO: remember that it means that we need proper extraction of args (pos or kwargs...)
        # possibly by extracting info from __call__ signature or process() if we modify from call to it

        # TODO:

        sample_ids_text = "@".join([str(x) for x in sorted(orig_sample_ids)])
        samples_ids_hash = hashlib.md5(sample_ids_text.encode("utf-8")).hexdigest()

        hash_filename = "samples_ids_hash@" + samples_ids_hash + ".pkl.gz"

        read_dirs = self._get_read_dirs()
        for curr_read_dir in read_dirs:
            fullpath_filename = os.path.join(curr_read_dir, "full_sets_info", hash_filename)
            if os.path.isfile(fullpath_filename):
                print(f"entire samples set {hash_filename} already cached. Found {os.path.abspath(fullpath_filename)}")
                return load_pickle(fullpath_filename)

        orig_sid_to_final = OrderedDict()
        for_global_storage = {"samples_cacher_instance": self}
        all_ans = run_multiprocessed(
            SamplesCacher._cache_worker,
            orig_sample_ids,
            workers=self._workers,
            copy_to_global_storage=for_global_storage,
            verbose=1,
            desc="caching",
        )

        for initial_sample_id, output_sample_ids in zip(orig_sample_ids, all_ans):
            orig_sid_to_final[initial_sample_id] = output_sample_ids

        write_dir = self._get_write_dir()
        set_info_dir = os.path.join(write_dir, "full_sets_info")
        os.makedirs(set_info_dir, exist_ok=True)
        fullpath_filename = os.path.join(set_info_dir, hash_filename)
        save_pickle_safe(orig_sid_to_final, fullpath_filename, compress=True)

        return orig_sid_to_final

    @staticmethod
    def get_final_sample_id_hash(sample_id):
        """
        sample_id is the final sample_id that came out of the pipeline
        note: our pipeline supports Ops returning None, thus, discarding a sample (in that case, it will not have any final sample_id),
        additionally, the pipeline may return *multiple* samples, each with their own sample_id
        """
        curr_sample_id_str = str(sample_id)  # TODO repr or str ?
        output_sample_hash = hashlib.md5(curr_sample_id_str.encode("utf-8")).hexdigest()
        ans = f"out_sample_id@{output_sample_hash}"
        return ans

    @staticmethod
    def get_orig_sample_id_hash(orig_sample_id):
        """
        orig_sample_id is the original sample_id that was provided, regardless if it turned out to become None, the same sample_id, or different sample_id(s)
        """
        orig_sample_id_str = str(orig_sample_id)
        if orig_sample_id_str.startswith("<") and orig_sample_id_str.endswith(">"):  # and '0x' in orig_sample_id_str
            # <__main__.SomeClass at 0x7fc3e6645e20>
            raise Exception(
                f"You must implement a proper __str__ for orig_sample_id. String representations like <__main__.SomeClass at 0x7fc3e6645e20> are not descriptibe enough and also not persistent between runs. Got: {orig_sample_id_str}"
            )
        ans = hashlib.md5(orig_sample_id_str.encode("utf-8")).hexdigest()
        ans = "out_info_for_orig_sample@" + ans
        return ans

    def get_orig_sample_id_from_final_sample_id(self, orig_sample_id):
        pass

    def load_sample(self, sample_id: Hashable, keys: Optional[Sequence[str]] = None):
        """
        :param sample_id: the sample_id of the sample to load
        :param keys: optionally, provide a subset of the keys to load in this sample.
        This is useful for speeding up loading.
        """

        sample_from_cache = self._load_sample_from_cache(sample_id, keys)
        audit_required = self._audit.update()

        if audit_required:
            initial_sample_id = get_initial_sample_id(sample_from_cache)
            fresh_sample = self._load_sample_using_pipeline(initial_sample_id, keys)
            fresh_sample = get_specific_sample_from_potentially_morphed(fresh_sample, sample_id)

            self._audit.audit(sample_from_cache, fresh_sample)

        return sample_from_cache

    def _load_sample_using_pipeline(self, sample_id: Hashable, keys: Optional[Sequence[str]] = None):
        sample_dict = create_initial_sample(sample_id)
        result_sample = self._pipeline(sample_dict)
        return result_sample

    def _load_sample_from_cache(self, sample_id: Hashable, keys: Optional[Sequence[str]] = None):
        """
        TODO: add comments
        """
        read_dirs = self._get_read_dirs()
        sample_hash = SamplesCacher.get_final_sample_id_hash(sample_id)

        for curr_read_dir in read_dirs:
            extension_less = os.path.join(curr_read_dir, sample_hash)
            if os.path.isfile(extension_less + ".pkl.gz"):
                loaded_sample = NDict(load_pickle(extension_less + ".pkl.gz"))
                if os.path.isfile(extension_less + ".hdf5"):
                    loaded_sample_hdf5_part = load_hdf5(extension_less + ".hdf5")
                    loaded_sample.merge(loaded_sample_hdf5_part)
                return loaded_sample

        raise Exception(f"Expected to find a cached sample for sample_id={sample_id} but could not find any!")

    @staticmethod
    def _cache_worker(orig_sample_id: Any):
        cacher = get_from_global_storage("samples_cacher_instance")
        ans = cacher._cache(orig_sample_id)
        return ans

    def _cache(self, orig_sample_id: Any):
        """
        :param orig_sample_id: the original sample id, which was provided as the input to the pipeline
        :param sample: the result of the pipeline - can be None if it was dropped, a dictionary in the typical standard case,
         and a list of dictionaries in case the sample was split into multiple samples (ops are allowed to do that during the static part of the processing)
        """

        write_dir = self._get_write_dir()
        os.makedirs(write_dir, exist_ok=True)
        read_dirs = self._get_read_dirs()

        was_processed_hash = SamplesCacher.get_orig_sample_id_hash(orig_sample_id)
        was_processed_fn = was_processed_hash + ".pkl"

        # checking in all read directories if information related to this sample(s) was already cached
        for curr_read_dir in read_dirs:
            fn = os.path.join(curr_read_dir, was_processed_fn)
            if os.path.isfile(fn):
                ans = load_pickle(fn)
                return ans

        result_sample = self._load_sample_using_pipeline(orig_sample_id)

        if isinstance(result_sample, dict):
            result_sample = [result_sample]

        if isinstance(result_sample, list):
            if 0 == len(result_sample):
                result_sample = None
            for s in result_sample:
                set_initial_sample_id(s, orig_sample_id)

        if not isinstance(result_sample, (list, dict, type(None))):
            raise Exception(
                f"Unsupported sample type, got {type(result_sample)}. Supported types are dict, list-of-dicts and None."
            )

        if result_sample is not None:
            output_info = []
            for curr_sample in result_sample:
                curr_sample_id = get_sample_id(curr_sample)
                output_info.append(curr_sample_id)
                output_sample_hash = SamplesCacher.get_final_sample_id_hash(curr_sample_id)

                requiring_hdf5_keys = _object_requires_hdf5_recurse(curr_sample)
                if len(requiring_hdf5_keys) > 0:
                    requiring_hdf5_dict = curr_sample.get_multi(requiring_hdf5_keys)
                    requiring_hdf5_dict = requiring_hdf5_dict.flatten()

                    hdf5_filename = os.path.join(write_dir, output_sample_hash + ".hdf5")
                    save_hdf5_safe(hdf5_filename, **requiring_hdf5_dict)

                    # remove all hdf5 entries from the sample_dict that will be pickled
                    for k in requiring_hdf5_dict:
                        _ = curr_sample.pop(k)

                save_pickle_safe(curr_sample, os.path.join(write_dir, output_sample_hash + ".pkl.gz"), compress=True)
        else:
            output_info = None
            # requiring_hdf5_keys = None

        save_pickle_safe(output_info, os.path.join(write_dir, was_processed_fn))
        return output_info


def _get_available_write_location(cache_dirs: List[str], max_allowed_used_space=0.95):
    """
    :param cache_dirs: write directories. Directories are checked in order that they are provided.
    :param max_allowed_used_space: set to a value between 0.0 to 1.0.
    a value of 0.95 means that once the available space is greater or equal to 95% of the the disk capacity,
    it will be considered full, and the next directory will be attempted.
    """

    for curr_loc in cache_dirs:
        if max_allowed_used_space is None:
            return curr_loc
        os.makedirs(curr_loc, exist_ok=True)
        drive_stats = psutil.disk_usage(curr_loc)
        actual_usage_part = drive_stats.percent / 100.0
        if actual_usage_part < max_allowed_used_space:
            return curr_loc

    raise Exception(
        "Could not find any location to write.\n"
        f"write_cache_locations={cache_dirs}\n"
        f"max_allowed_used_space={max_allowed_used_space}"
    )


def default_read_dirs_logic(cache_dirs: List[str]):
    return cache_dirs
