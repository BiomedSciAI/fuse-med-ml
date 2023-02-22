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

import fuse_examples.imaging.classification.stoic21.dataset as dataset

if "STOIC21_DATA_PATH" in os.environ:
    from fuse_examples.imaging.classification.stoic21.runner_stoic21 import (
        PATHS,
        TRAIN_COMMON_PARAMS,
        run_train,
        run_infer,
        run_eval,
        INFER_COMMON_PARAMS,
        EVAL_COMMON_PARAMS,
        DATASET_COMMON_PARAMS,
    )


def run_stoic21(root: str) -> None:
    model_dir = os.path.join(root, "model_dir")
    paths = {
        "model_dir": model_dir,
        "data_dir": PATHS["data_dir"],
        "cache_dir": os.path.join(root, "cache_dir"),
        "data_split_filename": os.path.join(root, "split.pkl"),
        "inference_dir": os.path.join(model_dir, "infer_dir"),
        "eval_dir": os.path.join(model_dir, "eval_dir"),
    }

    train_common_params = TRAIN_COMMON_PARAMS
    train_common_params["trainer.num_epochs"] = 2
    infer_common_params = INFER_COMMON_PARAMS

    analyze_common_params = EVAL_COMMON_PARAMS
    dataset_common_params = DATASET_COMMON_PARAMS

    GPU.choose_and_enable_multiple_gpus(1)

    Seed.set_seed(0, False)  # previous test (in the pipeline) changed the deterministic behavior to True
    train_dataset, infer_dataset = dataset.create_dataset(paths=paths, params=dataset_common_params)
    run_train(train_dataset, infer_dataset, paths, train_common_params)
    run_infer(infer_dataset, paths, infer_common_params)
    results = run_eval(paths, analyze_common_params)

    assert "metrics.auc" in results


@unittest.skipIf(
    "STOIC21_DATA_PATH" not in os.environ, "define environment variable 'STOIC21_DATA_PATH' to run this test"
)
class ClassificationStoic21TestCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_template(self):
        run_in_subprocess(run_stoic21, self.root, timeout=1800)

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main()
