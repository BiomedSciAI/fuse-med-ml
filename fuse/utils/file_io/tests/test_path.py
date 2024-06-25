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
from fuse.utils.file_io.file_io import save_hdf5_safe, load_hdf5
import numpy as np
import tempfile
import os
from fuse.utils.file_io import path


class TestPath(unittest.TestCase):
    """
    Test path.py
    """

    def setUp(self) -> None:
        pass

    def test_path_1(self) -> None:

        ans = path.add_base_prefix("/a/b/c/de/fg/banana.phone", "hohoho@")
        self.assertEqual(ans, "/a/b/c/de/fg/hohoho@banana.phone")

        ans = path.change_extension("/a/b/c/de/fg/123.txt", "7zip")
        self.assertEqual(ans, "/a/b/c/de/fg/123.7zip")

        ans = path.change_extension("/a/b/c/de/fg/123.456.txt", "7zip")
        self.assertEqual(ans, "/a/b/c/de/fg/123.456.7zip")

        ans = path.get_extension("/a/b/c/de/fg/123.456.7zip")
        self.assertEqual(ans, ".7zip")

        ans = path.remove_extension("/a/b/c/de/fg/123.456.7zip")
        self.assertEqual(ans, "/a/b/c/de/fg/123.456")

        ans = path.get_valid_filename("test 1 2 3 he^^llo")
        self.assertEqual(ans, "test_1_2_3_he^^llo")

    def tearDown(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
