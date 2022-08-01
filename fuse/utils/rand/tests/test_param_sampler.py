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


import random

from fuse.utils import Uniform, Choice, RandInt, RandBool, draw_samples_recursively, Seed


class TestParamSampler(unittest.TestCase):
    """
    Test ParamSampleBase sub classes
    """

    def test_uniform(self):
        Seed.set_seed(0)
        min = random.random() * 1000
        max = random.random() * 1000 + min
        uniform = Uniform(min, max)
        value = uniform.sample()

        # test range
        self.assertGreaterEqual(value, min)
        self.assertLessEqual(uniform.sample(), max)

        # test generate more than a single number
        self.assertNotEqual(value, uniform.sample())

        # test fixed per seed
        Seed.set_seed(1234)
        value0 = uniform.sample()
        Seed.set_seed(1234)
        value1 = uniform.sample()
        self.assertEqual(value0, value1)

    def test_randint(self):
        Seed.set_seed(0)
        min = random.randint(0, 1000)
        max = random.randint(0, 1000) + min
        randint = RandInt(min, max)
        value = randint.sample()

        # test range
        self.assertGreaterEqual(value, min)
        self.assertLessEqual(randint.sample(), max)

        # test generate more than a single number
        self.assertNotEqual(value, randint.sample())

        # test fixed per seed
        Seed.set_seed(1234)
        value0 = randint.sample()
        Seed.set_seed(1234)
        value1 = randint.sample()
        self.assertEqual(value0, value1)

    def test_randbool(self):
        Seed.set_seed(0)
        randbool = RandBool(0.5)
        value = randbool.sample()

        # test range
        self.assertIn(value, [True, False])

        # test generate more than a single number
        Seed.set_seed(0)
        values = [randbool.sample() for _ in range(4)]
        self.assertIn(True, values)
        self.assertIn(False, values)

        # test fixed per seed
        Seed.set_seed(1234)
        value0 = randbool.sample()
        Seed.set_seed(1234)
        value1 = randbool.sample()
        self.assertEqual(value0, value1)

        # test probs
        Seed.set_seed(0)
        randbool = RandBool(0.99)
        count = 0
        for _ in range(1000):
            if randbool.sample() == True:
                count += 1
        self.assertGreaterEqual(count, 980)

    def test_choice(self):
        Seed.set_seed(0)
        lst = list(range(1000))
        choice = Choice(lst)
        value = choice.sample()

        # test range
        self.assertIn(value, lst)

        # test generate more than a single number
        self.assertNotEqual(value, choice.sample())

        # test fixed per seed
        Seed.set_seed(1234)
        value0 = choice.sample()
        Seed.set_seed(1234)
        value1 = choice.sample()
        self.assertEqual(value0, value1)

        # test probs
        Seed.set_seed(0)
        probs = [0.01 / 999] * 1000
        probs[5] = 0.99
        choice = Choice(lst, probs)
        count = 0
        for _ in range(1000):
            if choice.sample() == 5:
                count += 1
        self.assertGreaterEqual(count, 980)

    def test_draw_samples_recursively(self):
        Seed.set_seed(0)
        a = {
            "a": 5,
            "b": [3, RandInt(1, 5), 9],
            "c": {"d": 3, "f": [1, 2, RandBool(0.5), {"h": RandInt(10, 15)}]},
            "e": {"g": Choice([6, 7, 8])},
        }
        b = draw_samples_recursively(a)

        self.assertEqual(a["a"], a["a"])
        self.assertEqual(b["b"][0], a["b"][0])
        self.assertEqual(b["b"][2], a["b"][2])
        self.assertEqual(b["c"]["d"], a["c"]["d"])
        self.assertEqual(b["c"]["f"][1], a["c"]["f"][1])
        self.assertIn(b["b"][1], [1, 2, 3, 4, 5])
        self.assertIn(b["c"]["f"][2], [True, False])
        self.assertIn(b["c"]["f"][3]["h"], [10, 11, 12, 13, 14, 15])
        self.assertIn(b["e"]["g"], [6, 7, 8])


if __name__ == "__main__":
    unittest.main()
