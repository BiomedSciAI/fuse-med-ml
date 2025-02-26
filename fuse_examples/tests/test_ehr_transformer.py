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

import os
import shutil
import tempfile
import unittest

import torch
import yaml
from omegaconf import DictConfig

from fuse.utils.multiprocessing.run_multiprocessed import run_in_subprocess
from fuse.utils.tests.decorators import skipIfMultiple
from fuse_examples.multimodality.ehr_transformer.main_train import main as main_train

if "CINC_TEST_DATA_PATH" in os.environ:
    TEST_PARAMS = {
        "CONFIG_PATH": os.path.dirname(__file__)
        + "/../multimodality/ehr_transformer/config.yaml",
        "TEST_DATA_DIR": os.environ["CINC_TEST_DATA_PATH"],
        "TEST_DATA_PICKL": None,
        "NAME": "ehr_transformer_test",
        "BATCH_SIZE": 30,
        "NUM_EPOCHS": 2,
    }


def init_test_environment(root: str) -> DictConfig:
    with open(TEST_PARAMS["CONFIG_PATH"], "r") as file:
        cfg = yaml.safe_load(file)

    # update confifuration relevant to the unitest TEST DATA folder
    cfg["root"] = root
    cfg["name"] = TEST_PARAMS["NAME"]
    cfg["data"]["batch_size"] = TEST_PARAMS["BATCH_SIZE"]
    cfg["data"]["dataset_cfg"]["raw_data_path"] = TEST_PARAMS["TEST_DATA_DIR"]
    cfg["data"]["dataset_cfg"]["raw_data_pkl"] = TEST_PARAMS["TEST_DATA_PICKL"]
    cfg["train"]["trainer_kwargs"]["max_epochs"] = TEST_PARAMS["NUM_EPOCHS"]

    return cfg


def run_ehr_transformer(cfg: DictConfig) -> None:
    try:
        main_train(cfg)

    except:
        raise Exception("Training EHR Transformer Failed.")


@skipIfMultiple(
    (
        "CINC_TEST_DATA_PATH" not in os.environ,
        "define environment variable 'CINC_TEST_DATA_PATH' to run this test",
    ),
    (not torch.cuda.is_available(), "No GPU is available"),
)
class EHRTransformerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.cfg = init_test_environment(self.root)

    def test_template(self) -> None:
        run_in_subprocess(run_ehr_transformer, self.cfg)

    def tearDown(self) -> None:
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
