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
import pathlib

import fuse.utils.gpu as GPU
from fuse_examples.imaging.classification.prostate_x.run_train_3dpatch import TRAIN_COMMON_PARAMS, train_template, infer_template, eval_template, INFER_COMMON_PARAMS, \
    EVAL_COMMON_PARAMS


class ClassificationProstateXTestCase(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

        root_path = self.root
        root_data = '/projects/msieve/MedicalSieve/PatientData/ProstateX/manifest-A3Y4AE4o5818678569166032044/' 
        self.paths = {'force_reset_model_dir': True,
         # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'model_dir': os.path.join(root_path, 'prostatex/my_model/'),
         'cache_dir': os.path.join(root_path, 'prostatex/my_cache/'),
         'inference_dir': os.path.join(root_path, 'prostatex/my_model/inference/'),
         'eval_dir': os.path.join(root_path,  'prostatex/my_model/eval/'),
         'data_dir': os.path.join(pathlib.Path(__file__).parent.resolve(), "../classification/prostate_x"),
         'prostate_data_path' : root_data,
         'ktrans_path': os.path.join(root_data, 'ProstateXKtrains-train-fixed/'),
         }


        self.train_common_params = TRAIN_COMMON_PARAMS

        self.infer_common_params = INFER_COMMON_PARAMS

        self.analyze_common_params = EVAL_COMMON_PARAMS

    @unittest.skip("Not ready yet")
    # TODO:
    #  1. Get path as an env variable
    #  2. modify the result value check 
    def test_template(self):
        num_gpus_allocated = GPU.choose_and_enable_multiple_gpus(1, use_cpu_if_fail=True)
        if num_gpus_allocated == 0:
            self.train_common_params['manager.train_params']['device'] = 'cpu'
        train_template(self.paths, self.train_common_params)
        infer_template(self.paths, self.infer_common_params)
        results = eval_template(self.paths, self.analyze_common_params)

        threshold = 0.98
        self.assertGreaterEqual(results['metrics.auc.macro_avg'], threshold)

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == '__main__':
    unittest.main()