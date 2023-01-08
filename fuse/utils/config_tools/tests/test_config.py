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

from fuse.utils.config_tools import Config
import os


class TestConfig(unittest.TestCase):
    def test_config_no_py_extension(self):
        """ """
        _curr_dir = os.path.dirname(os.path.abspath(__file__))
        _reference_ans = {"test": 240, "banana": 123, "dvivonim": 10}

        conf = Config()
        z_no_py_ext__internal_include = conf.load(
            {"test": 14},
            os.path.join(_curr_dir, "some_conf_internal_include"),
            {"test": 240},
        )

        z_with_ext__internal_include = conf.load(
            {"test": 14},
            os.path.join(_curr_dir, "some_conf_internal_include.py"),
            {"test": 240},
        )

        z_with_ext__external_include = conf.load(
            {"test": 14},
            os.path.join(_curr_dir, "base_conf_example.py"),
            os.path.join(_curr_dir, "some_conf_no_include.py"),
            {"test": 240},
        )

        z_with_ext__no_include = conf.load(
            {"test": 14},
            os.path.join(_curr_dir, "some_conf_no_include.py"),
            {"test": 240},
        )

        self.assertEqual(z_no_py_ext__internal_include, _reference_ans)
        self.assertEqual(z_no_py_ext__internal_include, z_with_ext__internal_include)
        self.assertEqual(z_no_py_ext__internal_include, z_with_ext__external_include)
        self.assertNotEqual(_reference_ans, z_with_ext__no_include)


if __name__ == "__main__":
    unittest.main()
