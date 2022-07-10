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

from typing import List, Optional, Union
import unittest

import pandas as pds
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.utils.collates import CollateDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.op_base import OpBase
from fuse.data import get_sample_id


class OpCustomCollateDefTest(OpBase):
    def __call__(self, sample_dict: dict, **kwargs) -> Union[None, dict, List[dict]]:
        if get_sample_id(sample_dict) == "a":
            sample_dict["data.partial"] = 1
        return sample_dict


class TestCollate(unittest.TestCase):
    def test_collate_default(self):
        # datainfo
        data = {
            "sample_id": ["a", "b", "c", "d", "e"],
            "data.values": [7, 4, 9, 2, 4],
            "data.nps": [np.array(4), np.array(2), np.array(5), np.array(1), np.array(4)],
            "data.torch": [torch.tensor(7), torch.tensor(4), torch.tensor(9), torch.tensor(2), torch.tensor(4)],
            "data.not_important": [12] * 5,
        }
        df = pds.DataFrame(data)

        # create simple pipeline
        op_df = OpReadDataframe(df)
        op_partial = OpCustomCollateDefTest()
        pipeline = PipelineDefault("test", [(op_df, {}), (op_partial, {})])

        # create dataset
        dataset = DatasetDefault(data["sample_id"], dynamic_pipeline=pipeline)
        dataset.create()

        # Use the collate function
        dl = DataLoader(
            dataset, 3, collate_fn=CollateDefault(skip_keys=["data.not_important"], raise_error_key_missing=False)
        )
        batch = next(iter(dl))

        # verify
        self.assertTrue("data.sample_id" in batch)
        self.assertListEqual(batch["data.sample_id"], ["a", "b", "c"])
        self.assertTrue((batch["data.values"] == torch.tensor([7, 4, 9])).all())
        self.assertTrue("data.nps" in batch)
        self.assertTrue((batch["data.nps"] == torch.stack([torch.tensor(4), torch.tensor(2), torch.tensor(5)])).all())
        self.assertTrue("data.torch" in batch)
        self.assertTrue((batch["data.torch"] == torch.stack([torch.tensor(7), torch.tensor(4), torch.tensor(9)])).all())
        self.assertTrue("data.partial" in batch)
        self.assertListEqual(batch["data.partial"], [1, None, None])
        self.assertFalse("data.not_important" in batch)

    def test_pad_all_tensors_to_same_size(self):
        a = torch.zeros((1, 1, 3))
        b = torch.ones((1, 2, 1))
        values = CollateDefault.pad_all_tensors_to_same_size([a, b])

        self.assertTrue((np.array(values.shape[1:]) == np.maximum(a.shape, b.shape)).all())
        self.assertTrue((values[1][:, :, :1] == b).all())
        self.assertTrue(values[1].sum() == b.sum())

    def test_pad_all_tensors_to_same_size_bs_1(self):
        a = torch.ones((1, 2, 1))
        values = CollateDefault.pad_all_tensors_to_same_size([a])
        self.assertTrue((values[0] == a).all())

    def test_pad_all_tensors_to_same_size_bs_3(self):
        a = torch.ones((1, 2, 3))
        b = torch.ones((3, 2, 1))
        c = torch.ones((1, 3, 2))
        values = CollateDefault.pad_all_tensors_to_same_size([a, b, c])
        self.assertListEqual(list(values.shape), [3, 3, 3, 3])


if __name__ == "__main__":
    unittest.main()
