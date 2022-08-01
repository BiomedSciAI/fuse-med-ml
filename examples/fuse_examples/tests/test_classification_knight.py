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
import pathlib
import shutil
import tempfile
import unittest
import os
from fuse.utils.file_io.file_io import create_dir
import sys
import wget
# add parent directory to path, so that 'knight' folder is treated as a module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from imaging.classification.knight.eval.eval import eval
from imaging.classification.knight.make_targets_file import make_targets_file
import imaging.classification.knight.baseline.fuse_baseline as baseline
from fuseimg.datasets.kits21 import KITS21

class KnightTestTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        # KITS21.download(self.root, cases=list(range(2)))

    def test_eval(self):
        dir_path = pathlib.Path(__file__).parent.resolve()
        target_filename = os.path.join(dir_path, "../imaging/classification/knight/eval/example/example_targets.csv")
        task1_prediction_filename = os.path.join(
            dir_path, "../imaging/classification/knight/eval/example/example_task1_predictions.csv"
        )
        task2_prediction_filename = os.path.join(
            dir_path, "../imaging/classification/knight/eval/example/example_task2_predictions.csv"
        )
        eval(
            target_filename=target_filename,
            task1_prediction_filename=task1_prediction_filename,
            task2_prediction_filename=task2_prediction_filename,
            output_dir=self.root,
        )

    def test_make_targets(self):
        dir_path = pathlib.Path(__file__).parent.resolve()
        data_path = os.path.join(self.root, "data")
        cache_path = os.path.join(self.root, "cache")
        output_filename = os.path.join(self.root, "output/validation_targets.csv")

        create_dir(data_path)
        create_dir(os.path.dirname(output_filename))
        shutil.copy(os.path.join(dir_path,"../imaging/classification/knight/eval/knight_test_labels.json"), data_path)
        make_targets_file(data_path=data_path, cache_path=cache_path, split=None, output_filename=output_filename)

    def test_train(self):
        os.environ["KNIGHT_DATA"] = "/projects/msieve/MedicalSieve/PatientData/KNIGHT/knight/data"
        os.environ["KNIGHT_CACHE"] = os.path.join(self.root, "train", "cache")
        os.environ["KNIGHT_RESULTS"] = os.path.join(self.root, "train", "results")
        config = """
        experiment_num : 0
        task_num : task_1 # task_1 or task_2 
        num_gpus : 1
        use_data : {"imaging": True, "clinical": True} 
        batch_size : 2
        resize_to : [70, 256, 256]
        num_epochs : 0
        learning_rate : 0.0001
        imaging_dropout : 0.5
        fused_dropout : 0.5

        task_1:
            num_classes : 2
            target_name : "data.gt.gt_global.task_1_label"
            target_metric : "validation.metrics.auc"
        """
        cfg_path = os.path.join(self.root,"test_cfg.yaml")
        with open(cfg_path, "w") as file:
            file.write(config)
        
        wget.download(
            "https://raw.github.com/neheller/KNIGHT/main/knight/data/knight.json",
            os.environ["KNIGHT_DATA"],
        )
        baseline.main(cfg_path)

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main()
