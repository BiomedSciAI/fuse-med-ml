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
from fuse.data.ops.op_base import op_call

import pandas as pd
from fuse.utils.ndict import NDict
from fuse.data.ops.ops_read import OpReadDataframe


class TestOpsRead(unittest.TestCase):
    def test_op_read_dataframe(self):
        """
        Test OpReadDataframe
        """
        data = {"sample_id": ["a", "b", "c", "d"], "data.value1": [10, 7, 3, 9], "data.value2": ["5", "4", "3", "2"]}
        df = pd.DataFrame(data)
        op = OpReadDataframe(data=df)
        sample_dict = NDict({"data": {"sample_id": "c"}})
        sample_dict = op_call(op, sample_dict, "id")
        self.assertEqual(sample_dict["data.value1"], 3)
        self.assertEqual(sample_dict["data.value2"], "3")

        op = OpReadDataframe(data=df, columns_to_extract=["sample_id", "data.value2"])
        sample_dict = NDict({"data": {"sample_id": "c"}})
        sample_dict = op_call(op, sample_dict, "id")
        self.assertFalse("data.value1" in sample_dict)
        self.assertEqual(sample_dict["data.value2"], "3")

        op = OpReadDataframe(
            data=df, columns_to_extract=["sample_id", "data.value2"], rename_columns={"data.value2": "data.value3"}
        )
        sample_dict = NDict({"data": {"sample_id": "c"}})
        sample_dict = op_call(op, sample_dict, "id")
        self.assertFalse("data.value1" in sample_dict)
        self.assertFalse("data.value2" in sample_dict)
        self.assertEqual(sample_dict["data.value3"], "3")


if __name__ == "__main__":
    unittest.main()
