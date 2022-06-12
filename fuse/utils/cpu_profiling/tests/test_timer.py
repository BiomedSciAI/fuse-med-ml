import unittest

from fuse.utils.cpu_profiling import Timer


class TestTimer(unittest.TestCase):
    def test_timer_1(self):
        """ """

        def foo(x):
            with Timer("banana"):
                for _ in range(10**5):
                    x = x * 2

        foo(20)


if __name__ == "__main__":
    unittest.main()
