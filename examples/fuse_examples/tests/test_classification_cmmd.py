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
from fuse_examples.imaging.classification.cmmd.runner import run_train, run_eval, run_infer
from omegaconf import DictConfig, OmegaConf
from fuse.utils import NDict
import unittest
import os
import tempfile
import shutil
import hydra
import sys
config = None

from fuse.utils.gpu import choose_and_enable_multiple_gpus

# @unittest.skip("Not ready yet")
# TODO:
# 1. Get the path to data as an env variable
# 2. Consider reducing the number of samples
class ClassificationMGCmmdTestCase(unittest.TestCase):
    def setUp(self):
        global config
        self.cfg = NDict(OmegaConf.to_container(config))
        print(self.cfg)
        
        # Path to the stored dataset location
        # dataset should be download from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508
        # download requires NBIA data retriever https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
        # put on the folliwing in the main folder  - 
        # 1. CMMD_clinicaldata_revision.csv which is a converted version of CMMD_clinicaldata_revision.xlsx 
        # 2. folder named CMMD which is the downloaded data folder


    def test_runner(self):
        # uncomment if you want to use specific gpus instead of automatically looking for free ones
        force_gpus = None  # [0]
        choose_and_enable_multiple_gpus(self.cfg["train.manager_train_params.num_gpus"], force_gpus=force_gpus)

        run_train(self.cfg["paths"] ,self.cfg["train"])
        run_infer(self.cfg["paths"] , self.cfg["infer"])
        results = run_eval(self.cfg["paths"] , self.cfg["infer"])

        threshold = 0.6
        self.assertGreaterEqual(results['metrics.auc'], threshold)

    # def tearDown(self):
    #     # Delete temporary directories
    #     # shutil.rmtree(self.working_dir)
        
@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    global config
    config = cfg
    unittest.main()
if __name__ == '__main__':
    main()

