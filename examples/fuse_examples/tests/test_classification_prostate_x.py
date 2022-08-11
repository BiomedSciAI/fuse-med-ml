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

import shutil
import tempfile
import unittest
import os
from fuse.utils.multiprocessing.run_multiprocessed import run_in_subprocess

from fuse.utils.rand.seed import Seed
import fuse.utils.gpu as GPU

from fuse_examples.imaging.classification.prostate_x.runner_prostate_x import (
    TRAIN_COMMON_PARAMS,
    INFER_COMMON_PARAMS,
    EVAL_COMMON_PARAMS,
    PATHS,
    run_train,
    run_infer,
    run_eval,
)


def run_prostate_x(root: str) -> None:

    model_dir = os.path.join(root, "model_dir")
    paths = {
        "model_dir": model_dir,
        "force_reset_model_dir": True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
        "data_dir": PATHS["data_dir"],
        "cache_dir": os.path.join(root, "cache_dir"),
        "data_split_filename": os.path.join(root, "split.pkl"),
        "inference_dir": os.path.join(model_dir, "infer_dir"),
        "eval_dir": os.path.join(model_dir, "eval_dir"),
    }

    train_common_params = TRAIN_COMMON_PARAMS
    train_common_params["trainer.num_epochs"] = 2
    infer_common_params = INFER_COMMON_PARAMS

    eval_common_params = EVAL_COMMON_PARAMS

    GPU.choose_and_enable_multiple_gpus(1)

    Seed.set_seed(0, False)  # previous test (in the pipeline) changed the deterministic behavior to True
    run_train(paths, train_common_params)
    run_infer(paths, infer_common_params)
    results = run_eval(paths, eval_common_params)

    assert "metrics.auc" in results


@unittest.skipIf(
    "PROSTATEX_DATA_PATH" not in os.environ, "define environment variable 'PROSTATEX_DATA_PATH' to run this test"
)
@unittest.skip("Takes too much time.")
class ClassificationProstateXTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_template(self):
        run_in_subprocess(run_prostate_x, self.root, timeout=1800)

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main()
