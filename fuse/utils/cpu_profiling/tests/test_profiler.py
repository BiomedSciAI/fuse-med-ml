import unittest

from fuse.utils.cpu_profiling import Profiler


class TestTimer(unittest.TestCase):
    def test_profiler_1(self):
        """ """
        from itertools import combinations

        def dummy_slow():
            N = 140  # 1000
            K = 2
            for _ in range(100):
                all_combs = combinations(range(N), K)
                all_combs = list(all_combs)
            return all_combs

        def dummy_quick():
            a = 1000
            a += 1
            return a

        def get_combs():
            for _ in range(10):
                x = dummy_slow()
                y = dummy_quick()

            return x, y

        with Profiler("banana"):
            get_combs()


if __name__ == "__main__":
    unittest.main()
