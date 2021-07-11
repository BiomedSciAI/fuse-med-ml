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

from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseUtilsHierarchicalDictTestCase(unittest.TestCase):

    def setUp(self):
        self.hierarchical_dict = {'a': 1, 'b': {'c': 2, 'd': 3}, 'c': 4}

    def test_get(self):
        self.assertEqual(FuseUtilsHierarchicalDict.get(self.hierarchical_dict, 'a'), 1)
        self.assertEqual(FuseUtilsHierarchicalDict.get(self.hierarchical_dict, 'b.c'), 2)
        self.assertEqual(FuseUtilsHierarchicalDict.get(self.hierarchical_dict, 'c'), 4)
        self.assertDictEqual(FuseUtilsHierarchicalDict.get(self.hierarchical_dict, 'b'), {'c': 2, 'd': 3})

    def test_set(self):
        FuseUtilsHierarchicalDict.set(self.hierarchical_dict, 'a', 7)
        self.assertEqual(self.hierarchical_dict['a'], 7)
        FuseUtilsHierarchicalDict.set(self.hierarchical_dict, 'a', 1)
        self.assertEqual(self.hierarchical_dict['a'], 1)

        FuseUtilsHierarchicalDict.set(self.hierarchical_dict, 'b.d', 5)
        self.assertEqual(self.hierarchical_dict['b']['d'], 5)
        FuseUtilsHierarchicalDict.set(self.hierarchical_dict, 'b.d', 3)
        self.assertEqual(self.hierarchical_dict['b']['d'], 3)

        FuseUtilsHierarchicalDict.set(self.hierarchical_dict, 'b.e.f.g', 9)
        self.assertEqual(self.hierarchical_dict['b']['e']['f']['g'], 9)
        del self.hierarchical_dict['b']['e']

    def test_pop(self):
        FuseUtilsHierarchicalDict.pop(self.hierarchical_dict, 'b.d')
        all_keys = FuseUtilsHierarchicalDict.get_all_keys(self.hierarchical_dict)
        self.assertSetEqual(set(all_keys), {'a', 'b.c', 'c'})

        FuseUtilsHierarchicalDict.pop(self.hierarchical_dict, 'c')
        all_keys = FuseUtilsHierarchicalDict.get_all_keys(self.hierarchical_dict)
        self.assertSetEqual(set(all_keys), {'a', 'b.c'})
        with self.assertRaises(KeyError):
            FuseUtilsHierarchicalDict.pop(self.hierarchical_dict, 'c')
        with self.assertRaises(KeyError):
            FuseUtilsHierarchicalDict.pop(self.hierarchical_dict, 'lala')

    def test_get_all_keys(self):
        all_keys = FuseUtilsHierarchicalDict.get_all_keys(self.hierarchical_dict)
        self.assertSetEqual(set(all_keys), {'a', 'b.c', 'b.d', 'c'})

    def test_is_in(self):
        self.assertTrue(FuseUtilsHierarchicalDict.is_in(self.hierarchical_dict, 'a'))
        self.assertTrue(FuseUtilsHierarchicalDict.is_in(self.hierarchical_dict, 'b.c'))
        self.assertTrue(FuseUtilsHierarchicalDict.is_in(self.hierarchical_dict, 'b.d'))
        self.assertTrue(FuseUtilsHierarchicalDict.is_in(self.hierarchical_dict, 'c'))
        self.assertFalse(FuseUtilsHierarchicalDict.is_in(self.hierarchical_dict, 'd'))
        self.assertFalse(FuseUtilsHierarchicalDict.is_in(self.hierarchical_dict, 'e'))

    def tearDown(self):
        delattr(self, 'hierarchical_dict')


if __name__ == '__main__':
    unittest.main()
