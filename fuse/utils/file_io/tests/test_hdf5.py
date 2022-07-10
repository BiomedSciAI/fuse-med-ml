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


class TestHDF5IO(unittest.TestCase):
    """
    Test HDF5 IO
    """

    def setUp(self):
        pass

    def _generate_test_data_1(self):
        Seed.set_seed(1337)
        ans = {
            "data.cc.img": np.random.rand(30, 200, 200),
            "data.cc.seg": np.random.randint(0, 16, size=(30, 200, 200)),
            "data.mlo.img": np.random.rand(30, 200, 200),
            "data.mlo.seg": np.random.randint(0, 16, size=(40, 100, 164)),
            "data.clinical_info_input": np.random.rand(1000),
        }
        return ans

    def test_object_requires_hdf5_recurse(self):
        data = self._generate_test_data_1()
        tmpdir = tempfile.gettempdir()
        filename = os.path.join(tmpdir, "test_hdf5_1.hdf5")

        save_hdf5_safe(filename, **data)
        loaded_hdf5_all = load_hdf5(filename)
        loaded_hdf5_partial = load_hdf5(
            filename,
            custom_extract={
                "data.cc.img": [slice(1, 10, 2), slice(None, 20), Ellipsis],
                "data.cc.seg": None,
                "data.mlo.img": 20,
                "data.clinical_info_input": None,
            },
        )

        self.assertEqual(len(loaded_hdf5_all.keys()), 5)
        self.assertAlmostEqual(loaded_hdf5_all["data.cc.img"].sum(), 599968.1459841608)
        self.assertEqual(loaded_hdf5_all["data.cc.seg"].sum(), 9005579)
        self.assertAlmostEqual(loaded_hdf5_all["data.mlo.img"].sum(), 599511.3034792768)

        self.assertEqual(len(loaded_hdf5_partial.keys()), 4)
        self.assertAlmostEqual(loaded_hdf5_partial["data.cc.img"].sum(), 9991.398330095684)
        self.assertEqual(loaded_hdf5_partial["data.cc.seg"].sum(), 9005579)
        self.assertAlmostEqual(loaded_hdf5_partial["data.mlo.img"].sum(), 20010.11375657579)
        self.assertAlmostEqual(loaded_hdf5_partial["data.clinical_info_input"].sum(), 507.3055890687598)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
