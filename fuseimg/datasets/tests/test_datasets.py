import os
import pathlib
import shutil
from tempfile import mkdtemp
import unittest
from fuse.data.utils.sample import get_sample_id
from fuse.utils.file_io.file_io import create_dir

from fuseimg.datasets.kits21 import KITS21
from fuseimg.datasets.stoic21 import STOIC21
from tqdm import trange
from testbook import testbook
from fuse.eval.metrics.stat.metrics_stat_common import MetricUniqueValues
from fuse.utils.multiprocessing.run_multiprocessed import get_from_global_storage, run_multiprocessed
from fuse.eval.evaluator import EvaluatorDefault

from fuseimg.datasets.isic import ISIC
from examples.fuse_examples.imaging.classification.isic.golden_members import FULL_GOLDEN_MEMBERS, TEN_GOLDEN_MEMBERS


notebook_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "../kits21_example.ipynb")


def ds_getitem(index: int):
    sample = get_from_global_storage("ds")[index].flatten()
    del sample["data.input.img"]
    return sample


class TestDatasets(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.kits21_cache_dir = mkdtemp(prefix="kits21_cache")
        self.kits21_data_dir = (
            os.environ["KITS21_DATA_PATH"] if "KITS21_DATA_PATH" in os.environ else mkdtemp(prefix="kits21_data")
        )

        self.stoic21_cache_dir = mkdtemp(prefix="stoic_cache")

        self.isic_cache_dir = mkdtemp(prefix="isic_cache")
        self.isic_data_dir = (
            os.environ["ISIC19_DATA_PATH"] if "ISIC19_DATA_PATH" in os.environ else mkdtemp(prefix="isic_data")
        )

    def test_kits21(self):
        KITS21.download(self.kits21_data_dir, cases=list(range(10)))

        create_dir(self.kits21_cache_dir)
        dataset = KITS21.dataset(
            data_path=self.kits21_data_dir,
            cache_dir=self.kits21_cache_dir,
            reset_cache=True,
            sample_ids=[f"case_{id:05d}" for id in range(10)],
        )
        self.assertEqual(len(dataset), 10)
        for sample_index in trange(10):
            sample = dataset[sample_index]
            self.assertEqual(get_sample_id(sample), f"case_{sample_index:05d}")

    @unittest.skipIf(
        "STOIC21_DATA_PATH" not in os.environ, "Expecting environment variable STOIC21_DATA_PATH to be defined"
    )
    def test_stoic21(self):
        data_path = os.environ["STOIC21_DATA_PATH"]
        sids = STOIC21.sample_ids(data_path)[:10]
        ds = STOIC21.dataset(sample_ids=sids, data_path=data_path, cache_dir=self.stoic21_cache_dir, reset_cache=True)

        metrics = {
            "age": MetricUniqueValues(key="data.input.age"),
            "gender": MetricUniqueValues(key="data.input.gender"),
            "thickness": MetricUniqueValues(key="data.metadata.SliceThickness"),
            "covid": MetricUniqueValues(key="data.gt.probCOVID"),
            "severe": MetricUniqueValues(key="data.gt.probSevere"),
        }
        evaluator = EvaluatorDefault()

        data_iter = run_multiprocessed(
            worker_func=ds_getitem,
            args_list=range(len(ds)),
            copy_to_global_storage={"ds": ds},
            workers=10,
            verbose=1,
            as_iterator=True,
        )

        results = evaluator.eval(ids=None, data=data_iter, metrics=metrics, id_key="data.sample_id")

        self.assertEqual(ds[0]["data.input.clinical"].shape[0], 8)
        self.assertTrue(5 in dict(results["metrics.age"]))

    def test_isic(self):

        create_dir(self.isic_cache_dir)
        dataset = ISIC.dataset(
            self.isic_data_dir, self.isic_cache_dir, train=True, reset_cache=True, samples_ids=TEN_GOLDEN_MEMBERS
        )
        self.assertEqual(len(dataset), 10)
        for sample_index in range(10):
            sample = dataset[sample_index]
            self.assertEqual(get_sample_id(sample), TEN_GOLDEN_MEMBERS[sample_index])

    @testbook(notebook_path, execute=range(0, 4), timeout=120)
    def test_basic(tb, self):
        tb.execute_cell([4, 5])

        tb.inject(
            """
            assert(np.max(my_dataset[0]['data.input.img'])>=0 and np.max(my_dataset[0]['data.input.img'])<=1)
            """
        )

    @testbook(notebook_path, execute=range(0, 4), timeout=240)
    def test_caching(tb, self):
        tb.execute_cell([9])

        tb.execute_cell([16, 17])
        tb.inject(
            """
            assert(isinstance(my_dataset[0]["data.gt.seg"], torch.Tensor))
            """
        )

    @testbook(notebook_path, execute=range(0, 4), timeout=240)
    def test_custom(tb, self):
        tb.execute_cell([25])

        tb.inject(
            """
            assert(my_dataset[0]["data.gt.seg"].shape[1:] == (4, 256, 256))
            """
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.kits21_cache_dir)
        if "KITS21_DATA_PATH" not in os.environ:
            shutil.rmtree(self.kits21_data_dir)

        shutil.rmtree(self.stoic21_cache_dir)

        shutil.rmtree(self.isic_cache_dir)
        if "ISIC19_DATA_PATH" not in os.environ:
            shutil.rmtree(self.isic_data_dir)

        super().tearDown()


if __name__ == "__main__":
    unittest.main()
