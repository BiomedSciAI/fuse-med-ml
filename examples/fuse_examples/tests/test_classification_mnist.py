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

from fuse_examples.imaging.classification.mnist.run_mnist import (
    EVAL_COMMON_PARAMS,
    INFER_COMMON_PARAMS,
    TRAIN_COMMON_PARAMS,
    run_train,
    run_infer,
    run_eval,
)


class ClassificationMnistTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        model_dir = os.path.join(self.root, "model_dir")
        self.paths = {
            "model_dir": model_dir,
            "force_reset_model_dir": True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
            "cache_dir": os.path.join(self.root, "cache_dir"),
            "inference_dir": os.path.join(model_dir, "infer_dir"),
            "eval_dir": os.path.join(model_dir, "eval_dir"),
        }

        self.train_common_params = TRAIN_COMMON_PARAMS

        self.infer_common_params = INFER_COMMON_PARAMS

        self.eval_common_params = EVAL_COMMON_PARAMS

    @run_in_subprocess()
    def test_template(self):
        Seed.set_seed(0, False)  # previous test (in the pipeline) changed the deterministic behavior to True
        run_train(self.paths, self.train_common_params)
        run_infer(self.paths, self.infer_common_params)
        results = run_eval(self.paths, self.eval_common_params)

        threshold = 0.95
        self.assertGreaterEqual(results["metrics.auc.macro_avg"], threshold)

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
