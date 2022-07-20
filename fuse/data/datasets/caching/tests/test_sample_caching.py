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

"""

import unittest

from fuse.utils.rand.seed import Seed
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data import get_sample_id, create_initial_sample
import numpy as np
import tempfile
import os
from fuse.data.ops.op_base import OpBase
from typing import List, Union
from fuse.data.datasets.caching.samples_cacher import SamplesCacher

from fuse.utils.ndict import NDict


class OpFakeLoad(OpBase):
    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, **kwargs) -> Union[None, dict, List[dict]]:
        sid = get_sample_id(sample_dict)
        if "case_1" == sid:
            sample_dict.merge(_generate_sample_1())
        elif "case_2" == sid:
            sample_dict.merge(_generate_sample_2())
        elif "case_3" == sid:
            return None
        elif "case_4" == sid:
            sample_1 = create_initial_sample("case_4", "case_4_subcase_1")
            sample_1.merge(_generate_sample_1(41))

            sample_2 = create_initial_sample("case_4", "case_4_subcase_2")
            sample_2.merge(_generate_sample_2(42))

            return [sample_1, sample_2]
        else:
            raise Exception(f"unfamiliar sample_id: {sid}")
        return sample_dict


class TestSampleCaching(unittest.TestCase):
    """
    Test sample caching
    """

    def setUp(self):
        pass

    def test_cache_samples(self):
        orig_sample_ids = ["case_1", "case_2", "case_3", "case_4"]
        tmpdir = tempfile.mkdtemp()
        cache_dirs = [
            os.path.join(tmpdir, "cache_a"),
            os.path.join(tmpdir, "cache_b"),
        ]

        pipeline_desc = [
            (OpFakeLoad(), {}),
        ]
        pl = PipelineDefault("example_pipeline", pipeline_desc)

        cacher = SamplesCacher("unittests_cache", pl, cache_dirs, restart_cache=True)

        cacher.cache_samples(orig_sample_ids)

        sample = cacher.load_sample("case_1")
        sample = cacher.load_sample("case_2")
        sample = cacher.load_sample("case_4_subcase_1")
        sample = cacher.load_sample("case_4_subcase_2")
        # sample = cacher.load_sample('case_3') #isn't supposed to work
        # sample = cacher.load_sample('case_4') #isn't supposed to work

        banana = 123

    def test_same_uniquely_named_cache_and_multiple_pipeline_hashes(self):
        orig_sample_ids = ["case_1", "case_2", "case_3", "case_4"]
        tmpdir = tempfile.mkdtemp()
        cache_dirs = [
            os.path.join(tmpdir, "cache_c"),
            os.path.join(tmpdir, "cache_d"),
        ]

        pipeline_desc = [
            (OpFakeLoad(), {}),
        ]
        pl = PipelineDefault("example_pipeline", pipeline_desc)
        cacher = SamplesCacher("unittests_cache", pl, cache_dirs, restart_cache=True)

        cacher.cache_samples(orig_sample_ids)

        ### now, we modify the pipeline and we DO NOT set restart_cache, to verify an exception is thrown
        pipeline_desc = [
            (OpFakeLoad(), {}),
            (OpFakeLoad(), {}),  ###just doubled it to change the pipeline hash
        ]
        pl = PipelineDefault("example_pipeline", pipeline_desc)
        self.assertRaises(Exception, SamplesCacher, "unittests_cache", pl, cache_dirs, restart_cache=False)

    def tearDown(self):
        pass


def _generate_sample_1(seed=1337):
    Seed.set_seed(seed)
    sample = NDict(
        dict(
            data=dict(
                cc=dict(
                    img=np.random.rand(30, 200, 200),
                    seg=np.random.randint(0, 16, size=(30, 200, 200)),
                    dicom_tags=[10, 13, 40, "banana"],
                ),
                mlo=dict(
                    img=np.random.rand(30, 200, 200),
                    seg=np.random.randint(0, 16, size=(40, 100, 164)),
                    dicom_tags=[100, 130, 400, "banana123"],
                ),
                gt_labels_style_1=[1, 3, 100, 12],
                gt_labels_style_2=np.array([3, 4, 10, 12]),
                clinical_info_input=np.random.rand(1000),
            )
        )
    )
    return sample


def _generate_sample_2(seed=1234):
    Seed.set_seed(seed)
    sample = NDict(
        dict(
            data=dict(
                cc=dict(
                    img=np.random.rand(10, 100, 100),
                    seg=np.random.randint(0, 16, size=(10, 100, 10)),
                    dicom_tags=[20, 23, 60, "bananaphone"],
                ),
                mlo=dict(
                    img=np.random.rand(30, 200, 200),
                    seg=np.random.randint(0, 16, size=(40, 100, 164)),
                    dicom_tags=[12, 13, 40, "porcupine123"],
                ),
                gt_labels_style_1=[5, 2, 13, 16],
                gt_labels_style_2=np.array([8, 14, 11, 1]),
                clinical_info_input=np.random.rand(90),
            )
        )
    )
    return sample


if __name__ == "__main__":
    unittest.main()
