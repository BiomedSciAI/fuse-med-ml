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

===============================

The most minimal MNIST classifier implementation that demonstrate end to end training using Fuse that we could make.
There is a lot more that one could do on top of this, including inference, evaluation, model checkpointing, etc. But
the goal of this script is simply to train a classifier and show how to wrap a regular pytorch model so it can be
trained "fuse" style. Meaning it accesses data using keys to a batch_dict. It also demonstrates how easy it is to
load datasets from fuse.
"""

import copy
import os
from typing import OrderedDict

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy

from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault

from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.lightning.pl_module import LightningModuleDefault

from fuseimg.datasets.mnist import MNIST
from examples.fuse_examples.imaging.classification.mnist import lenet

ROOT = "_examples/mnist"  # TODO: fill path here
model_dir = os.path.join(ROOT, "model_dir")
PATHS = {
    "model_dir": model_dir,
    "cache_dir": os.path.join(ROOT, "cache_dir"),
    "inference_dir": os.path.join(model_dir, "infer_dir"),
    "eval_dir": os.path.join(model_dir, "eval_dir"),
}
train_params = {
    "data.batch_size": 100,
    "data.train_num_workers": 8,
    "data.validation_num_workers": 8,
    "trainer.num_epochs": 12,
    "trainer.num_devices": 1,
    "trainer.accelerator": "gpu",
    "trainer.strategy": "dp",
    "trainer.ckpt_path": None,
    "opt.lr": 1e-4,
    "opt.weight_decay": 0.001,
}


class FuseLitLenet(LightningModuleDefault):
    def __init__(self):
        super().__init__()
        # wrap basic torch model to automatically read inputs from batch_dict and save its outputs to batch_dict
        self._model = ModelWrapSeqToDict(
            model=lenet.LeNet(),
            model_inputs=["data.image"],
            post_forward_processing_function=lambda logits: (logits, F.softmax(logits, dim=1)),
            model_outputs=["model.logits.classification", "model.output.classification"],
        )
        self._losses = {
            "cls_loss": LossDefault(
                pred="model.logits.classification", target="data.label", callable=F.cross_entropy, weight=1.0
            ),
        }
        self._train_metrics = OrderedDict(
            [
                ("operation_point", MetricApplyThresholds(pred="model.output.classification")),  # will apply argmax
                ("accuracy", MetricAccuracy(pred="results:metrics.operation_point.cls_pred", target="data.label")),
            ]
        )
        self._validation_metrics = copy.deepcopy(self._train_metrics)
        optimizer = optim.Adam(
            self._model.parameters(), lr=train_params["opt.lr"], weight_decay=train_params["opt.weight_decay"]
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")
        self._optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)


def run_train(paths: dict, train_params: dict):
    train_dataset = MNIST.dataset(PATHS["cache_dir"], train=True)
    validation_dataset = MNIST.dataset(PATHS["cache_dir"], train=False)
    model = FuseLitLenet()

    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.label",
        num_balanced_classes=10,
        batch_size=train_params["data.batch_size"],
        balanced_class_weights=None,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=train_params["data.train_num_workers"],
    )

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=train_params["data.batch_size"],
        collate_fn=CollateDefault(),
        num_workers=train_params["data.validation_num_workers"],
    )

    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train_params["trainer.num_epochs"],
        accelerator=train_params["trainer.accelerator"],
        strategy=train_params["trainer.strategy"],
        devices=train_params["trainer.num_devices"],
        auto_select_gpus=True,
    )

    # train
    pl_trainer.fit(model, train_dataloader, validation_dataloader, ckpt_path=train_params["trainer.ckpt_path"])


if __name__ == "__main__":
    run_train(paths=PATHS, train_params=train_params)
