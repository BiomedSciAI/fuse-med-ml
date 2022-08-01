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


from fuse.utils.ndict import NDict
import numpy
import torch
import copy


class TestNDict(unittest.TestCase):
    """
    Test NDict class
    """

    def setUp(self):
        self.nested_dict = NDict({"a": 1, "b": {"c": 2, "d": 3}, "c": 4})

    def test_get(self):
        self.assertEqual(self.nested_dict["a"], 1)
        self.assertEqual(self.nested_dict["b.c"], 2)
        self.assertEqual(self.nested_dict["c"], 4)
        self.assertDictEqual(self.nested_dict["b"], {"c": 2, "d": 3})

    def test_set(self):
        self.nested_dict["a"] = 7
        self.assertEqual(self.nested_dict["a"], 7)
        self.nested_dict["a"] = 1
        self.assertEqual(self.nested_dict["a"], 1)

        self.nested_dict["b.d"] = 5
        self.assertEqual(self.nested_dict["b"]["d"], 5)
        self.nested_dict["b.d"] = 3
        self.assertEqual(self.nested_dict["b"]["d"], 3)

        self.nested_dict["b.e.f.g"] = 9
        self.assertEqual(self.nested_dict["b"]["e"]["f"]["g"], 9)
        del self.nested_dict["b"]["e"]

    def test_pop(self):
        self.nested_dict.pop("b.d")
        all_keys = self.nested_dict.keypaths()
        self.assertSetEqual(set(all_keys), {"a", "b.c", "c"})

        self.nested_dict.pop("c")
        all_keys = self.nested_dict.keypaths()
        self.assertSetEqual(set(all_keys), {"a", "b.c"})
        with self.assertRaises(KeyError):
            self.nested_dict.pop("c")
        with self.assertRaises(KeyError):
            self.nested_dict.pop("lala")

    def test_keypaths(self):
        all_keys = self.nested_dict.keypaths()
        self.assertSetEqual(set(all_keys), {"a", "b.c", "b.d", "c"})

    def test_is_in(self):
        self.assertTrue("a" in self.nested_dict)
        self.assertTrue("b.c" in self.nested_dict)
        self.assertTrue("b.d" in self.nested_dict)
        self.assertTrue("c" in self.nested_dict)
        self.assertFalse("d" in self.nested_dict)
        self.assertFalse("e" in self.nested_dict)

    def test_apply_on_all(self):
        nested_dict_copy = self.nested_dict.clone()

        def plus_one(val: int) -> int:
            return val + 1

        nested_dict_copy.apply_on_all(plus_one)

        # verify values
        all_keys = sorted(self.nested_dict.keypaths())
        all_keys_2 = sorted(nested_dict_copy.keypaths())
        self.assertListEqual(all_keys, all_keys_2)
        for key in all_keys:
            self.assertEqual(self.nested_dict[key] + 1, nested_dict_copy[key])

    def test_flatten(self):
        flat_dict = self.nested_dict.flatten()

        # verify
        all_keys = sorted(self.nested_dict.keypaths())
        flat_keys = sorted(flat_dict.keys())
        self.assertListEqual(all_keys, flat_keys)
        for key in flat_keys:
            self.assertEqual(self.nested_dict[key], flat_dict[key])

    def test_indices(self):
        nested_dict = NDict(
            {
                "a": numpy.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                "b": {"c": torch.tensor([[10, 11, 12], [13, 14, 15], [16, 17, 18]]), "d": [0, 1, 2], "f": 4},
            }
        )
        indices = numpy.array([True, False, True])
        nested_dict_indices = nested_dict.indices(indices)

        # verify
        self.assertTrue((nested_dict_indices["a"] == numpy.array([[0, 1, 2], [6, 7, 8]])).all())
        self.assertTrue((nested_dict_indices["b.c"] == torch.tensor([[10, 11, 12], [16, 17, 18]])).all())
        self.assertTrue((nested_dict_indices["b.d"] == [0, 2]))
        self.assertTrue((nested_dict_indices["b.f"] == 4))

    def tearDown(self):
        delattr(self, "nested_dict")


if __name__ == "__main__":
    unittest.main()
