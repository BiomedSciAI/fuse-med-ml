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

from fuse.utils.utils_gpu import FuseUtilsGPU
from fuse_examples.classification.mnist.runner import TRAIN_COMMON_PARAMS, run_train, run_infer, run_analyze, INFER_COMMON_PARAMS, \
    ANALYZE_COMMON_PARAMS


class ClassificationMnistTestCase(unittest.TestCase):

    def setUp(self):

        self.root = tempfile.mkdtemp()

        self.paths = {
            'model_dir': os.path.join(self.root, 'mnist/model_dir'),
            'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
            'cache_dir': os.path.join(self.root, 'mnist/cache_dir'),
            'inference_dir': os.path.join(self.root, 'mnist/infer_dir'),
            'analyze_dir': os.path.join(self.root, 'mnist/analyze_dir')}


        self.train_common_params = TRAIN_COMMON_PARAMS

        self.infer_common_params = INFER_COMMON_PARAMS

        self.analyze_common_params = ANALYZE_COMMON_PARAMS

    def test_template(self):
        num_gpus_allocated = FuseUtilsGPU.choose_and_enable_multiple_gpus(1, use_cpu_if_fail=True)
        if num_gpus_allocated == 0:
            self.train_common_params['manager.train_params']['device'] = 'cpu'
        run_train(self.paths, self.train_common_params)
        run_infer(self.paths, self.infer_common_params)
        results = run_analyze(self.paths, self.analyze_common_params)

        threshold = 0.98
        self.assertGreaterEqual(results['auc']['macro_avg'], threshold)

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == '__main__':
    unittest.main()