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
import pandas as pd
import torch

from fuse.data.processor.processor_dataframe import FuseProcessorDataFrame


class FuseProcessorDataFrameTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        pass

    def test_all_columns(self):
        df = pd.DataFrame({"desc": ['one', 'two', 'three'], "int_val": [4, 5, 6], "string_val": ['val1', 'val2', 'val3']})
        proc = FuseProcessorDataFrame(data=df, sample_desc_column='desc')

        self.assertDictEqual(proc('one'), {'int_val': 4, 'string_val':'val1'})
        self.assertDictEqual(proc('two'), {'int_val': 5, 'string_val':'val2'})
        self.assertDictEqual(proc('three'), {'int_val': 6, 'string_val':'val3'})

    def test_rename(self):
        df = pd.DataFrame({"desc": ['one', 'two', 'three'], "int_val": [4, 5, 6], "string_val": ['val1', 'val2', 'val3']})
        proc = FuseProcessorDataFrame(data=df, sample_desc_column='desc', rename_columns={'int_val': 'new_name'})

        self.assertDictEqual(proc('one'), {'new_name': 4, 'string_val':'val1'})
        self.assertDictEqual(proc('two'), {'new_name': 5, 'string_val':'val2'})
        self.assertDictEqual(proc('three'), {'new_name': 6, 'string_val':'val3'})

    def test_specific_columns(self):
        df = pd.DataFrame({"desc": ['one', 'two', 'three'], "int_val": [4, 5, 6], "string_val": ['val1', 'val2', 'val3']})
        proc = FuseProcessorDataFrame(data=df, sample_desc_column='desc', rename_columns={'int_val': 'new_name'},
                                      columns_to_extract=['string_val', 'desc'])

        self.assertDictEqual(proc('one'), {'string_val':'val1'})
        self.assertDictEqual(proc('two'), {'string_val':'val2'})
        self.assertDictEqual(proc('three'), {'string_val':'val3'})

    def test_tensors(self):
        df = pd.DataFrame({"desc": ['one', 'two', 'three'], "int_val": [4, 5, 6], "string_val": ['val1', 'val2', 'val3']})
        proc = FuseProcessorDataFrame(data=df, sample_desc_column='desc', columns_to_tensor=['int_val', 'invalid_column'])

        self.assertDictEqual(proc('one'), {'int_val': torch.tensor(4, dtype=torch.int64), 'string_val':'val1'})
        self.assertDictEqual(proc('two'), {'int_val': torch.tensor(5, dtype=torch.int64), 'string_val':'val2'})
        self.assertDictEqual(proc('three'), {'int_val': torch.tensor(6, dtype=torch.int64), 'string_val':'val3'})

    def test_tensors_with_types(self):
        df = pd.DataFrame({"desc": ['one', 'two', 'three'], "int_val": [4, 5, 6], "float_val": [4.1, 5.3, 6.5],
                           "string_val": ['val1', 'val2', 'val3']})
        proc = FuseProcessorDataFrame(data=df, sample_desc_column='desc',
                                      columns_to_tensor={'int_val': torch.int8, 'float_val': torch.float64})

        self.assertDictEqual(proc('one'), {'int_val': torch.tensor(4, dtype=torch.int8),
                                           'float_val': torch.tensor(4.1, dtype=torch.float64), 'string_val':'val1'})
        self.assertDictEqual(proc('two'), {'int_val': torch.tensor(5, dtype=torch.int8),
                                           'float_val': torch.tensor(5.3, dtype=torch.float64), 'string_val':'val2'})
        self.assertDictEqual(proc('three'), {'int_val': torch.tensor(6, dtype=torch.int8),
                                           'float_val': torch.tensor(6.5, dtype=torch.float64), 'string_val':'val3'})


    def test_tensors_non_existing(self):
        df = pd.DataFrame({"desc": ['one', 'two', 'three'], "int_val": [4, 5, 6], "string_val": ['val1', 'val2', 'val3']})
        proc = FuseProcessorDataFrame(data=df, sample_desc_column='desc', columns_to_tensor=['invalid_column'])

        self.assertDictEqual(proc('one'), {'int_val': 4, 'string_val':'val1'})
        self.assertDictEqual(proc('two'), {'int_val': 5, 'string_val':'val2'})
        self.assertDictEqual(proc('three'), {'int_val': 6, 'string_val':'val3'})


    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()