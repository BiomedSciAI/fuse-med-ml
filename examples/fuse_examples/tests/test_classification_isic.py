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

import logging
import unittest
import os
import tempfile
import shutil
from fuse.utils.multiprocessing.run_multiprocessed import run_in_subprocess
from fuse.utils.utils_logger import fuse_logger_end

from fuse_examples.imaging.classification.isic.runner import TRAIN_COMMON_PARAMS, INFER_COMMON_PARAMS, EVAL_COMMON_PARAMS,\
                                                                 run_train, run_infer, run_eval, PATHS

import fuse.utils.gpu as GPU
from fuse.utils.rand.seed import Seed
from fuseimg.datasets.isic import ISIC


class ClassificationISICTestCase(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

        self.paths = {
            'model_dir': os.path.join(self.root, 'isic/model_dir'),
            'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
            'data_dir': PATHS["data_dir"],
            'cache_dir': os.path.join(self.root, 'isic/cache_dir'),
            'data_split_filename': os.path.join(self.root, 'split.pkl'),
            'inference_dir': os.path.join(self.root, 'isic/infer_dir'),
            'eval_dir': os.path.join(self.root, 'isic/analyze_dir')}
        self.train_common_params = TRAIN_COMMON_PARAMS
        self.train_common_params['manager.train_params']['num_epochs'] = 15

        self.train_common_params = TRAIN_COMMON_PARAMS
        self.infer_common_params = INFER_COMMON_PARAMS
        self.eval_common_params = EVAL_COMMON_PARAMS

        self.isic = ISIC(data_path = self.paths['data_dir'], 
                    cache_path = self.paths['cache_dir'],
                    val_portion = 0.3)
        self.isic.download()

    @run_in_subprocess
    def test_runner(self):
        num_gpus_allocated = GPU.choose_and_enable_multiple_gpus(1, use_cpu_if_fail=True)
        if num_gpus_allocated == 0:
            self.train_common_params['manager.train_params']['device'] = 'cpu'

        Seed.set_seed(0, False) # previous test (in the pipeline) changed the deterministic behavior to True
        run_train(self.paths, self.train_common_params, self.isic)
        run_infer(self.paths, self.infer_common_params, self.isic)
        results = run_eval(self.paths, self.eval_common_params)

        threshold = 0.65
        self.assertGreaterEqual(results['metrics.auc'], threshold)

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == '__main__':
    unittest.main()



