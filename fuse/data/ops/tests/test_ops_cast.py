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
import unittest

from typing import List
from fuse.data.ops.op_base import op_call, op_reverse
from fuse.utils.ndict import NDict
import pandas as pd
import torch
import numpy as np
from fuse.data.ops.ops_cast import OpToNumpy, OpToTensor


class TestOpsCast(unittest.TestCase):
    def test_op_to_tensor(self):
        """
        Test OpToTensor __call__ and reverse
        """
        op = OpToTensor()
        sample = NDict(
            {
                "sample_id": 7,
                "values": {
                    "val_np": np.array([7, 8, 9]),
                    "val_torch": torch.tensor([1, 2, 3]),
                    "val_int": 3,
                    "val_float": 3.5,
                    "str": "hi!",
                },
            }
        )

        sample = op_call(op, sample, "_.test_id", key="values.val_np")
        self.assertIsInstance(sample["values.val_np"], torch.Tensor)
        self.assertTrue((sample["values.val_np"] == torch.tensor([7, 8, 9])).all())
        self.assertIsInstance(sample["values.val_int"], int)

        sample = op_call(op, sample, "_.test_id", key=["values.val_torch", "values.val_float"])
        self.assertIsInstance(sample["values.val_torch"], torch.Tensor)
        self.assertIsInstance(sample["values.val_float"], torch.Tensor)
        self.assertTrue((sample["values.val_torch"] == torch.tensor([1, 2, 3])).all())
        self.assertEqual(sample["values.val_float"], torch.tensor(3.5))
        self.assertIsInstance(sample["values.val_int"], int)

        sample = op_reverse(
            op, sample, key_to_follow="values.val_np", key_to_reverse="values.val_np", op_id="_.test_id"
        )
        self.assertIsInstance(sample["values.val_np"], np.ndarray)

    def test_op_to_numpy(self):
        """
        Test OpToNumpy __call__ and reverse
        """
        op = OpToNumpy()
        sample = NDict(
            {
                "sample_id": 7,
                "values": {
                    "val_np": np.array([7, 8, 9]),
                    "val_torch": torch.tensor([1, 2, 3]),
                    "val_int": 3,
                    "val_float": 3.5,
                    "str": "hi!",
                },
            }
        )

        sample = op_call(op, sample, "_.test_id", key="values.val_torch")
        self.assertIsInstance(sample["values.val_torch"], np.ndarray)
        self.assertTrue((sample["values.val_torch"] == np.array([1, 2, 3])).all())
        self.assertIsInstance(sample["values.val_int"], int)

        sample = op_call(op, sample, "_.test_id", key=["values.val_np", "values.val_float"])
        self.assertIsInstance(sample["values.val_np"], np.ndarray)
        self.assertIsInstance(sample["values.val_float"], np.ndarray)
        self.assertTrue((sample["values.val_np"] == np.array([7, 8, 9])).all())
        self.assertEqual(sample["values.val_float"], np.array(3.5))
        self.assertIsInstance(sample["values.val_int"], int)

        sample = op_reverse(
            op, sample, key_to_follow="values.val_torch", key_to_reverse="values.val_torch", op_id="_.test_id"
        )
        self.assertIsInstance(sample["values.val_torch"], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
