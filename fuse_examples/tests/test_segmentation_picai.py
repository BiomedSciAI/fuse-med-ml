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
# FIXME: data_package
import multiprocessing
from fuse.utils import NDict
import unittest
import shutil
import tempfile
import os
from fuse.utils.gpu import choose_and_enable_multiple_gpus
from fuse.utils.multiprocessing.run_multiprocessed import run_in_subprocess

import hydra
from hydra import compose, initialize


# os env variable PICAI_DATA_PATH is a path to the stored dataset location
# dataset should be download from https://zenodo.org/record/6517398#.ZGEHVXZBxD8
# labels should be downloaded from https://github.com/DIAGNijmegen/picai_labels
# folder named PICAI which is the downloaded data folder with partition for images and labels
if "PICAI_DATA_PATH" in os.environ:
    from fuse_examples.imaging.segmentation.picai.runner import (
        run_train,
        run_infer,
        run_eval,
    )


def run_picai(root: str) -> None:
    initialize(config_path="../imaging/segmentation/picai/conf", job_name="test_app")
    cfg = compose(
        config_name="config_template", overrides=["paths.working_dir=" + root]
    )
    print(str(cfg))

    cfg = NDict(hydra.utils.instantiate(cfg))
    cfg["train.run_sample"] = 100
    cfg["train.trainer.num_epochs"] = 1
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    choose_and_enable_multiple_gpus(cfg["train.trainer.devices"], force_gpus=force_gpus)

    run_train(cfg["paths"], cfg["train"])
    run_infer(cfg["infer"], cfg["paths"], cfg["train"])
    # analyze - skipping as it crushes without metrics
    results = run_eval(cfg["paths"], cfg["infer"])
    print(results)


@unittest.skipIf(
    "PICAI_DATA_PATH" not in os.environ,
    "define environment variable 'PICAI_DATA_PATH' to run this test",
)
class SegmentationPICAITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()

    def test_runner(self) -> None:
        run_in_subprocess(run_picai, self.root)

    def tearDown(self) -> None:
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main()
