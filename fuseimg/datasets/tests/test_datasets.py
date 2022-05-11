import os
import pathlib
import shutil
from tempfile import gettempdir, mkdtemp
import unittest
from fuse.data.utils.sample import get_sample_id
from fuse.utils.file_io.file_io import create_dir


from fuseimg.datasets.kits21 import KITS21
from tqdm import trange
from testbook import testbook

from fuseimg.datasets.skin_lesion import SkinLesion

notebook_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "../kits21_example.ipynb")

class TestDatasets(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        tmpdir = gettempdir()

        self.kits21_cache_dir = mkdtemp(prefix="kits21_cache")
        self.kits21_data_dir = mkdtemp(prefix="kits21_data")
        # self.skin_lesion_cache_dir = mkdtemp(prefix="skin_lesion_cache")
        self.skin_lesion_cache_dir = os.path.join(tmpdir, 'skin_lesion_cache')
        # self.skin_lesion_data_dir = mkdtemp(prefix="skin_lesion_data")
        self.skin_lesion_data_dir = os.path.join(tmpdir, 'skin_lesion_data_dir')

    def test_kits32(self):
        KITS21.download(self.kits21_data_dir, cases=list(range(10)))

        create_dir(self.kits21_cache_dir)
        dataset = KITS21.dataset(data_path=self.kits21_data_dir, cache_dir=self.kits21_cache_dir, reset_cache=True, sample_ids=[f"case_{id:05d}" for id in range(10)])
        self.assertEqual(len(dataset), 10)
        for sample_index in trange(10):
            sample = dataset[sample_index]
            self.assertEqual(get_sample_id(sample), f"case_{sample_index:05d}")

    def test_skin_lesion(self):
        SkinLesion.download(self.skin_lesion_data_dir)

        create_dir(self.skin_lesion_cache_dir)
        dataset = SkinLesion.dataset(data_path=self.skin_lesion_data_dir, cache_dir=self.skin_lesion_cache_dir,
                                     reset_cache=True, sample_ids=SkinLesion.TEN_GOLDEN_MEMBERS)
        self.assertEqual(len(dataset), 10)
        for sample_index in range(10):
            sample = dataset[sample_index]
            self.assertEqual(get_sample_id(sample), SkinLesion.TEN_GOLDEN_MEMBERS[sample_index])

    @testbook(notebook_path, execute=range(0,4), timeout=120)
    def test_basic(tb, self):
        tb.execute_cell([4,5])

        tb.inject(
            """
            assert(np.max(my_dataset[0]['data.input.img'])>=0 and np.max(my_dataset[0]['data.input.img'])<=1)
            """
        )
    
    @testbook(notebook_path, execute=range(0,4), timeout=120)
    def test_caching(tb, self):
        tb.execute_cell([9])

        tb.execute_cell([16,17])
        tb.inject(
            """
            assert(isinstance(my_dataset[0]["data.gt.seg"], torch.Tensor))
            """
        )
    
    @testbook(notebook_path, execute=range(0,4), timeout=120)
    def test_custom(tb, self):
        tb.execute_cell([25])

        tb.inject(
            """
            assert(my_dataset[0]["data.gt.seg"].shape[1:] == (4, 256, 256))
            """
        )

    
    def tearDown(self) -> None:
        shutil.rmtree(self.kits21_cache_dir)
        shutil.rmtree(self.kits21_data_dir)

        super().tearDown()

    
    
if __name__ == '__main__':
    unittest.main()
