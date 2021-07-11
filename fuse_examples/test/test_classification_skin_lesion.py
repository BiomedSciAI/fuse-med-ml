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

import unittest
import os
import tempfile
import shutil

from fuse_examples.classification.skin_lesion.runner import TRAIN_COMMON_PARAMS, \
    INFER_COMMON_PARAMS, ANALYZE_COMMON_PARAMS, run_train, run_analyze, run_infer

from fuse.utils.utils_gpu import FuseUtilsGPU


class ClassificationSkinLesionTestCase(unittest.TestCase):

    def setUp(self):
        self.train_common_params = TRAIN_COMMON_PARAMS
        self.train_common_params['data.year'] = '2016'
        self.train_common_params['manager.train_params']['num_epochs'] = 5

        # Path to save model
        self.ROOT = tempfile.mkdtemp()
        # Path to store the data
        self.ROOT_DATA = tempfile.mkdtemp(prefix="skin_data")
        # Name of the experiment
        EXPERIMENT = 'InceptionResnetV2_2016_test'
        # Path to cache data
        self.CACHE_PATH = tempfile.mkdtemp(prefix="cache_data")
        # Name of the cached data folder
        EXPERIMENT_CACHE = 'ISIC_' + self.train_common_params['data.year']

        self.paths = {'data_dir': self.ROOT_DATA,
                      'model_dir': os.path.join(self.ROOT, EXPERIMENT, 'model_dir'),
                      'force_reset_model_dir': True,
                      'cache_dir': os.path.join(self.CACHE_PATH, EXPERIMENT_CACHE + '_cache_dir'),
                      'inference_dir': os.path.join(self.ROOT, EXPERIMENT, 'infer_dir'),
                      'analyze_dir': os.path.join(self.ROOT, EXPERIMENT, 'analyze_dir')}

        self.infer_common_params = INFER_COMMON_PARAMS
        self.infer_common_params['data.year'] = self.train_common_params['data.year']

        self.analyze_common_params = ANALYZE_COMMON_PARAMS
        self.analyze_common_params['data.year'] = self.train_common_params['data.year']

    def test_runner(self):
        num_gpus_allocated = FuseUtilsGPU.choose_and_enable_multiple_gpus(1, use_cpu_if_fail=True)
        if num_gpus_allocated == 0:
            self.train_common_params['manager.train_params']['device'] = 'cpu'

        run_train(self.paths, self.train_common_params)
        run_infer(self.paths, self.infer_common_params)
        results = run_analyze(self.paths, self.analyze_common_params)

        threshold = 0.65
        self.assertGreaterEqual(results['auc']['macro_avg'], threshold)

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.ROOT)
        shutil.rmtree(self.ROOT_DATA)
        shutil.rmtree(self.CACHE_PATH)


if __name__ == '__main__':
    unittest.main()

