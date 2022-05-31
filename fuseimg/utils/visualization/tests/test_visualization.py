import os
import pathlib
import shutil
import unittest
from tqdm import trange
from testbook import testbook

notebook_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "../visualization_example.ipynb")

class TestVis(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    @testbook(notebook_path, execute=range(0,4), timeout=120)
    def test_basic(tb, self):
        tb.execute_cell([1,2,3])
    
    @testbook(notebook_path, execute=range(0,4), timeout=120)
    def test_caching(tb, self):
        tb.execute_cell([4])

    
    @testbook(notebook_path, execute=range(0,4), timeout=120)
    def test_custom(tb, self):
        tb.execute_cell([5])

    
    def tearDown(self) -> None:
        super().tearDown()

    
    
if __name__ == '__main__':
    unittest.main()
