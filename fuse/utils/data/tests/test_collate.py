import unittest

from fuse.utils.data import CollateToBatchList


class TestTimer(unittest.TestCase):
    def test_collate(self):
        """ """

        x = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
        ref = {"a": [1, 3, 5], "b": [2, 4, 6]}
        col = CollateToBatchList(x)
        y = col(x)
        self.assertEqual(y.to_dict(), ref)


if __name__ == "__main__":
    unittest.main()
