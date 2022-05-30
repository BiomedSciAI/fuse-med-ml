import os
import pathlib
import shutil
import unittest
from tqdm import trange
from testbook import testbook

notebook_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "../visualization_example.ipynb")

class TestDatasets(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    @testbook(notebook_path, execute=range(0,4), timeout=120)
    def test_basic(tb, self):
        tb.execute_cell([1,2,3])

        # tb.inject(
        #     """
        #     assert(np.max(my_dataset[0]['data.input.img'])>=0 and np.max(my_dataset[0]['data.input.img'])<=1)
        #     """
        # )
    
    @testbook(notebook_path, execute=range(0,4), timeout=120)
    def test_caching(tb, self):
        tb.execute_cell([4])
        # tb.inject(
        #     """
        #     assert(isinstance(my_dataset[0]["data.gt.seg"], torch.Tensor))
        #     """
        # )
    
    @testbook(notebook_path, execute=range(0,4), timeout=120)
    def test_custom(tb, self):
        tb.execute_cell([5])

        # tb.inject(
        #     """
        #     assert(my_dataset[0]["data.gt.seg"].shape[1:] == (4, 256, 256))
        #     """
        # )

    
    def tearDown(self) -> None:
        # shutil.rmtree(self.kits21_cache_dir)
        # shutil.rmtree(self.kits21_data_dir)

        super().tearDown()

    
    
if __name__ == '__main__':
    unittest.main()
