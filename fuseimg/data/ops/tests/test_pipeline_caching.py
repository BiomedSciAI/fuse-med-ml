import unittest
import os
import tempfile


from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher

from fuseimg.datasets.kits21 import KITS21


class TestPipelineCaching(unittest.TestCase):
    def test_basic_1(self):
        """
        Test basic imaging ops
        """
        tmpdir = tempfile.mkdtemp()
        kits_dir = os.path.join(tmpdir, "kits21")
        cases = [100, 150, 200]
        KITS21.download(kits_dir, cases)

        static_pipeline = KITS21.static_pipeline(kits_dir)
        dynamic_pipeline = KITS21.dynamic_pipeline()

        cache_dirs = [
            os.path.join(tmpdir, "cache_a"),
            os.path.join(tmpdir, "cache_b"),
        ]

        cacher = SamplesCacher("fuseimg_ops_testing_cache", static_pipeline, cache_dirs)

        sample_ids = [f"case_{_:05}" for _ in cases]
        ds = DatasetDefault(
            sample_ids,
            static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
            cacher=cacher,
        )

        ds.create()


if __name__ == "__main__":
    unittest.main()
