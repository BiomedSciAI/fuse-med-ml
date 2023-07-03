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
from fuse.utils.multiprocessing.run_multiprocessed import run_in_subprocess

from fuse_examples.imaging.classification.isic.isic_runner import (
    TRAIN_COMMON_PARAMS,
    INFER_COMMON_PARAMS,
    EVAL_COMMON_PARAMS,
    run_train,
    run_infer,
    run_eval,
)
from fuse_examples.imaging.classification.isic.golden_members import FULL_GOLDEN_MEMBERS

import fuse.utils.gpu as GPU
from fuse.utils.rand.seed import Seed


def run_isic(
    root: str,
    train_common_params: dict,
    infer_common_params: dict,
    eval_common_params: dict,
) -> None:
    model_dir = os.path.join(root, "model_dir")
    paths = {
        "model_dir": model_dir,
        "data_dir": os.environ["ISIC19_DATA_PATH"]
        if "ISIC19_DATA_PATH" in os.environ
        else os.path.join(root, "isic/data_dir"),
        "cache_dir": os.path.join(root, "cache_dir"),
        "inference_dir": os.path.join(model_dir, "infer_dir"),
        "eval_dir": os.path.join(model_dir, "eval_dir"),
        "data_split_filename": os.path.join(root, "split.pkl"),
    }

    dd = paths["data_dir"]
    print(f"data_dir={dd}")

    # Must use GPU due a long running time
    GPU.choose_and_enable_multiple_gpus(1, use_cpu_if_fail=False)

    Seed.set_seed(
        0, False
    )  # previous test (in the pipeline) changed the deterministic behavior to True

    run_train(paths, train_common_params)
    run_infer(paths, infer_common_params)
    results = run_eval(paths, eval_common_params)

    assert "metrics.auc" in results


class ClassificationISICTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.train_common_params = TRAIN_COMMON_PARAMS
        self.train_common_params["trainer.num_epochs"] = 1
        self.train_common_params["data.samples_ids"] = FULL_GOLDEN_MEMBERS

        self.infer_common_params = INFER_COMMON_PARAMS
        self.eval_common_params = EVAL_COMMON_PARAMS

    def test_runner_transformer(self) -> None:
        # Define model parameters
        token_dim = 768
        self.train_common_params["model_type"] = "Transformer"
        self.train_common_params["model"] = dict(
            token_dim=token_dim,
            projection_kwargs=dict(
                image_shape=[300, 300], patch_shape=[30, 30], channels=3
            ),
            transformer_kwargs=dict(
                depth=12,
                heads=12,
                mlp_dim=token_dim * 4,
                dim_head=64,
                dropout=0.0,
                emb_dropout=0.0,
            ),
        )

        # Reflect model choice in the infer stage
        self.infer_common_params["model_type"] = self.train_common_params["model_type"]
        self.infer_common_params["model"] = self.train_common_params["model"]
        run_in_subprocess(
            run_isic,
            self.root,
            self.train_common_params,
            self.infer_common_params,
            self.eval_common_params,
        )

    def test_runner_resnet18(self) -> None:
        # Define model parameters for two CNN backbone types
        self.train_common_params["model_type"] = "CNN"
        self.train_common_params["model"] = dict(
            dropout_rate=0.5,
            layers_description=(256,),
            tabular_data_inputs=[("data.input.clinical.all", 19)],
            tabular_layers_description=(128,),
            backbone_type="Resnet18",
        )

        # Reflect model choice in the infer stage
        self.infer_common_params["model_type"] = self.train_common_params["model_type"]
        self.infer_common_params["model"] = self.train_common_params["model"]
        run_in_subprocess(
            run_isic,
            self.root,
            self.train_common_params,
            self.infer_common_params,
            self.eval_common_params,
        )

    def test_runner_inception(self) -> None:
        # Define model parameters for two CNN backbone types
        self.train_common_params["model_type"] = "CNN"
        self.train_common_params["model"] = dict(
            dropout_rate=0.5,
            layers_description=(256,),
            tabular_data_inputs=[("data.input.clinical.all", 19)],
            tabular_layers_description=(128,),
            backbone_type="InceptionResnetV2",
        )

        # Reflect model choice in the infer stage
        self.infer_common_params["model_type"] = self.train_common_params["model_type"]
        self.infer_common_params["model"] = self.train_common_params["model"]
        run_in_subprocess(
            run_isic,
            self.root,
            self.train_common_params,
            self.infer_common_params,
            self.eval_common_params,
        )

    def tearDown(self) -> None:
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main()
