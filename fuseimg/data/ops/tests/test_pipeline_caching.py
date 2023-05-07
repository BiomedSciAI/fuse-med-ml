import unittest
import os
import tempfile


from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher

from fuseimg.datasets.isic import ISIC


class TestPipelineCaching(unittest.TestCase):
    def test_basic_1(self) -> None:
        """
        Test basic imaging ops
        """
        tmpdir = tempfile.mkdtemp()
        isic_dir = os.environ["ISIC19_DATA_PATH"]

        static_pipeline = ISIC.static_pipeline(isic_dir)

        cache_dirs = [
            os.path.join(tmpdir, "cache_a"),
            os.path.join(tmpdir, "cache_b"),
        ]

        cacher = SamplesCacher("fuseimg_ops_testing_cache", static_pipeline, cache_dirs)

        sample_ids = ["ISIC_0072637", "ISIC_0072638", "ISIC_0072639"]

        ds = DatasetDefault(
            sample_ids,
            static_pipeline,
            cacher=cacher,
        )

        ds.create()


if __name__ == "__main__":
    unittest.main()
