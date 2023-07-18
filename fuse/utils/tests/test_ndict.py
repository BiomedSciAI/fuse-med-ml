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


class TestNDict(unittest.TestCase):
    """
    Test NDict class
    """

    def setUp(self) -> None:
        self.nested_dict = NDict({"a": 1, "b": {"c": 2, "d": 3}, "c": 4})

    def test_get(self) -> None:
        self.assertEqual(self.nested_dict["a"], 1)
        self.assertEqual(self.nested_dict["b.c"], 2)
        self.assertEqual(self.nested_dict["c"], 4)
        self.assertDictEqual(self.nested_dict["b"].to_dict(), {"c": 2, "d": 3})

    def test_set(self) -> None:
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

    def test_pop(self) -> None:
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

    def test_keypaths(self) -> None:
        all_keys = self.nested_dict.keypaths()
        self.assertSetEqual(set(all_keys), {"a", "b.c", "b.d", "c"})

        ndict = NDict()
        ndict["a.b.c"] = "d"
        ndict["a.b.c.d"] = "e"
        self.assertSetEqual(set(ndict.keypaths()), {"a.b.c", "a.b.c.d"})

    def test_top_level_keys(self) -> None:
        ndict = NDict()
        ndict["a.b.c"] = 11
        ndict["a.d"] = 13
        ndict["e.f"] = 17
        ndict["g.h.i"] = 19

        self.assertSetEqual(set(ndict.top_level_keys()), {"a", "e", "g"})

    def test_is_in(self) -> None:
        self.assertTrue("a" in self.nested_dict)
        self.assertTrue("b.c" in self.nested_dict)
        self.assertTrue("b.d" in self.nested_dict)
        self.assertTrue("c" in self.nested_dict)
        self.assertFalse("d" in self.nested_dict)
        self.assertFalse("e" in self.nested_dict)

        ndict = NDict({"a": 1, "b.c": 2})
        self.assertTrue("a" in ndict)
        self.assertTrue("b" in ndict)
        self.assertTrue("b.c" in ndict)
        self.assertFalse("c" in ndict)
        self.assertFalse("42" in ndict)

    def test_apply_on_all(self) -> None:
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

    def test_flatten(self) -> None:
        flat_dict = self.nested_dict.flatten()

        # verify
        all_keys = sorted(self.nested_dict.keypaths())
        flat_keys = sorted(flat_dict.keypaths())
        self.assertListEqual(all_keys, flat_keys)
        for key in flat_keys:
            self.assertEqual(self.nested_dict[key], flat_dict[key])

    def test_indices(self) -> None:
        nested_dict = NDict(
            {
                "a": numpy.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                "b": {
                    "c": torch.tensor([[10, 11, 12], [13, 14, 15], [16, 17, 18]]),
                    "d": [0, 1, 2],
                    "f": 4,
                },
            }
        )
        indices = numpy.array([True, False, True])
        nested_dict_indices = nested_dict.indices(indices)

        # verify
        self.assertTrue(
            (nested_dict_indices["a"] == numpy.array([[0, 1, 2], [6, 7, 8]])).all()
        )
        self.assertTrue(
            (
                nested_dict_indices["b.c"] == torch.tensor([[10, 11, 12], [16, 17, 18]])
            ).all()
        )
        self.assertTrue((nested_dict_indices["b.d"] == [0, 2]))
        self.assertTrue((nested_dict_indices["b.f"] == 4))

    def test_get_sub_dict(self) -> None:
        # set ndict
        ndict = NDict()
        ndict["a.b.c.c1"] = "x1"
        ndict["a.b.c.c2"] = "x2"
        ndict["a.b.c.c3"] = "x3"

        # verify
        self.assertDictEqual(
            ndict.get_sub_dict("a").to_dict(),
            {"b.c.c1": "x1", "b.c.c2": "x2", "b.c.c3": "x3"},
        )
        self.assertDictEqual(
            ndict.get_sub_dict("a.b").to_dict(),
            {"c.c1": "x1", "c.c2": "x2", "c.c3": "x3"},
        )
        self.assertDictEqual(
            ndict.get_sub_dict("a.b.c").to_dict(), {"c1": "x1", "c2": "x2", "c3": "x3"}
        )
        self.assertEqual(ndict.get_sub_dict("a.b.c.c1"), None)
        self.assertEqual(ndict.get_sub_dict("a.q.q"), None)

    def test_get_closest_keys(self) -> None:
        # set ndict
        ndict = NDict()
        ndict["a.b.c"] = 42

        # verify
        self.assertEqual(ndict.get_closest_keys("a")[0], "a")
        self.assertEqual(ndict.get_closest_keys("a.b")[0], "a.b")
        self.assertEqual(ndict.get_closest_keys("a.b.c", 1)[0], "a.b.c")
        self.assertEqual(ndict.get_closest_keys("a.bb", 2)[0], "a.b")

    def test_delete(self) -> None:
        # set ndict
        ndict = NDict()
        ndict["a.b"] = 42
        ndict["a.c"] = 42

        # delete
        del ndict["a.b"]

        # verify deletion and raised error
        with self.assertRaises(KeyError):
            del ndict["a.b"]
        self.assertDictEqual(ndict.to_dict(), {"a.c": 42})

        # case: delete both value and sub-dict
        ndict = NDict()
        ndict["a.b"] = 23
        ndict["a.b.c"] = 42
        del ndict["a.b"]

        self.assertDictEqual(ndict.to_dict(), {})

    def test_merge(self) -> None:
        # set ndict
        ndict1 = NDict()
        ndict2 = NDict()
        ndict1["a.b.c"] = None
        ndict2["a.b.d"] = None

        ndict1.merge(ndict2)
        sub_dict = ndict1.get_sub_dict("a.b")

        self.assertTrue(len(ndict1) == 2)
        self.assertDictEqual(sub_dict.to_dict(), {"c": None, "d": None})

    def test_nested(self) -> None:
        """
        basic test that tests the case when we give the constructor a nested dictionary as dict_like input
        """

        nested_dict = {"a": {"b": {"c": 42}}}
        ndict = NDict(nested_dict)

        self.assertEqual(ndict["a.b.c"], 42)

    def test_get_tree(self) -> None:
        """
        basic test to check that print_tree() function runs

        results:
        --- e -> 10
        --- a
        ------ d -> 23
        ------ b
        --------- c -> 42
        """

        ndict = NDict()
        ndict["a.b.c"] = 42
        ndict["a.d"] = 23
        ndict["e"] = 10

        gts = [
            "--- e -> 10\n--- a\n------ d -> 23\n------ b\n--------- c -> 42",
            "--- e -> 10\n--- a\n------ b\n--------- c -> 42\n------ d -> 23",
            "--- a\n------ b\n--------- c -> 42\n------ d -> 23\n--- e -> 10",
            "--- a\n------ d -> 23\n------ b\n--------- c -> 42\n--- e -> 10",
        ]
        ans = ndict.get_tree(print_values=True)
        self.assertIn(ans, gts)
        print(ans)

    def test_unflatten(self) -> None:
        ndict = NDict()
        ndict["a.b.c"] = 42
        ndict["a.b.d"] = 23

        self.assertDictEqual(ndict.unflatten()["a"]["b"], {"c": 42, "d": 23})
        self.assertDictEqual(ndict["a"].unflatten()["b"], {"c": 42, "d": 23})

    def tearDown(self) -> None:
        delattr(self, "nested_dict")


if __name__ == "__main__":
    unittest.main()
