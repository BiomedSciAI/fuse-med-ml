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

import fuse.data
import pkg_resources  # part of setuptools


class TestVersion(unittest.TestCase):
    def test_version(self):
        """
        Make sure data version equal to the installed version
        """
        pass
        # FIXME: uncomment when fixed in jenkins
        # version = pkg_resources.require("fuse-med-ml-data")[0].version
        # self.assertEqual(fuse.data.__version__, version)


if __name__ == "__main__":
    unittest.main()
