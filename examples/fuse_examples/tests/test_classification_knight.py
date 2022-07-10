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
import wget


# FIXME: data_package
# from fuse_examples.imaging.classification.knight.eval.eval import eval
# from fuse_examples.imaging.classification.knight.make_targets_file import make_targets_file
# import fuse_examples.imaging.classification.knight.baseline.fuse_baseline as baseline


@unittest.skip("FIXME: data_package")
class KnightTestTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

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
        split = os.path.join(dir_path, "../imaging/classification/knight/baseline/splits_final.pkl")
        output_filename = os.path.join(self.root, "output/validation_targets.csv")

        create_dir(os.path.join(data_path, "knight", "data"))
        create_dir(os.path.dirname(output_filename))
        wget.download(
            "https://raw.github.com/neheller/KNIGHT/main/knight/data/knight.json",
            os.path.join(data_path, "knight", "data"),
        )
        make_targets_file(data_path=data_path, cache_path=cache_path, split=split, output_filename=output_filename)

    @unittest.skip("Not ready yet")
    # TODOs: set KNIGHT data
    # 1 Set 'KNIGHT_DATA' ahead (and not in the test)
    # 2, Add code that skip test if this var wasn't set
    # 2. Modify main() to support overriding the arguments and override number of epochs to 2 (and maybe number of samples)
    # 3. Use and test make predictions (inference script)
    def test_train(self):
        os.environ["KNIGHT_DATA"] = "/projects/msieve/MedicalSieve/PatientData/KNIGHT"
        os.environ["KNIGHT_CACHE"] = os.path.join(self.root, "train", "cache")
        os.environ["KNIGHT_RESULTS"] = os.path.join(self.root, "train", "results")
        baseline.main()

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main()
