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

from fuse.utils.utils_misc import get_pretty_dataframe


class FuseUtilsHierarchicalDictTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_normal_strings(self):
        df = pd.DataFrame({'A': ['abc', 'de', 'efg'],
                           'B': ['a', 'abcde', 'abc'],
                           'C': [1, 2.5, 1.5]})

        col_width = 25
        df_as_string = get_pretty_dataframe(df)

        self.assertTrue('abcd' in df_as_string)
        self.assertTrue(f'|{" " * (col_width - 2)}1.0|\n' in df_as_string)
        self.assertTrue(f'| abcde{" " * (col_width - 5)}|' in df_as_string)
        self.assertTrue(f"|\n{'-' * (col_width + 2) * 3}\n" in df_as_string)

    def test_long_strings(self):
        df = pd.DataFrame({'A': ['abc', 'de', 'very long column name should be longer than 25'],
                           'B': ['a', 'abcde', 'abc'],
                           'C': [1, 2.5, 1.5]})

        col_width = len('very long column name should be longer than 25')
        df_as_string = get_pretty_dataframe(df)

        self.assertTrue('abcd' in df_as_string)
        self.assertTrue(f'|{" " * (col_width - 2)}1.0|\n' in df_as_string)
        self.assertTrue(f'| abcde{" " * (col_width - 5)}|' in df_as_string)
        self.assertTrue(f"|\n{'-' * (col_width + 2) * 3}\n" in df_as_string)

    def test_long_headers(self):
        df = pd.DataFrame({'very long column name should be longer than 25': ['abc', 'de', 'abcd'],
                           'B': ['a', 'abcde', 'abc'],
                           'C': [1, 2.5, 1.5]})

        col_width = len('very long column name should be longer than 25')
        df_as_string = get_pretty_dataframe(df)

        self.assertTrue('abcd' in df_as_string)
        self.assertTrue(f'|{" " * (col_width - 2)}1.0|\n' in df_as_string)
        self.assertTrue(f'| abcde{" " * (col_width - 5)}|' in df_as_string)
        self.assertTrue(f"|\n{'-' * (col_width + 2) * 3}\n" in df_as_string)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
