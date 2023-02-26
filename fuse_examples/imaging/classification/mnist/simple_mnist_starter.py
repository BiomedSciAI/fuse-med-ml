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
from typing import OrderedDict, Any, Tuple

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
from fuse_examples.imaging.classification.mnist import lenet

import torch

## Paths and Hyperparameters ############################################
ROOT = "_examples/mnist"  # TODO: fill path here
MODEL_DIR = os.path.join(ROOT, "model_dir")
PATHS = {
    "model_dir": MODEL_DIR,
    "cache_dir": os.path.join(ROOT, "cache_dir"),
}
TRAIN_PARAMS = {
    "data.batch_size": 100,
    "data.train_num_workers": 8,
    "data.validation_num_workers": 8,
    "trainer.num_epochs": 2,
    "trainer.num_devices": 1,
    "trainer.accelerator": "gpu",
    "trainer.strategy": "dp",
    "trainer.ckpt_path": None,
    "opt.lr": 1e-4,
    "opt.weight_decay": 0.001,
}


def perform_softmax(logits: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    cls_preds = F.softmax(logits, dim=1)
    return logits, cls_preds


## Model Definition #####################################################
class FuseLitLenet(LightningModuleDefault):
    def __init__(self, model_dir: str, lr: float, weight_decay: float):  # paths, train_params):
        """
        Initialize the model. We inheret from LightningModuleDefault to get functions necessary for lightning trainer.
        We also wrap the torch model so that it can train using keys from the fuse NDict batch dict.

        :param model_dir: location for checkpoints and logs.
        :param lr: learning rate for optimizer
        :param weight_decay: weight decay for Adam optimizer. This is a form of regularization that penalizes for learning large weights.
        """

        # wrap basic torch model to automatically read inputs from batch_dict and save its outputs to batch_dict
        model = ModelWrapSeqToDict(
            model=lenet.LeNet(),
            model_inputs=["data.image"],
            post_forward_processing_function=perform_softmax,
            model_outputs=["model.logits.classification", "model.output.classification"],
        )

        # losses and metrics to track
        losses = {
            "cls_loss": LossDefault(
                pred="model.logits.classification", target="data.label", callable=F.cross_entropy, weight=1.0
            ),
        }
        train_metrics = OrderedDict(
            [
                ("operation_point", MetricApplyThresholds(pred="model.output.classification")),  # will apply argmax
                ("accuracy", MetricAccuracy(pred="results:metrics.operation_point.cls_pred", target="data.label")),
            ]
        )
        validation_metrics = copy.deepcopy(train_metrics)  # same as train metrics

        # optimizer and learning rate scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")
        optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

        # initialize LightningModuleDefault with our module, losses, etc so that we can use the functions of LightningModuleDefault
        super().__init__(
            model_dir=model_dir,
            model=model,
            losses=losses,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            optimizers_and_lr_schs=optimizers_and_lr_schs,
        )


## Training #############################################################
def run_train(paths: dict, train_params: dict):

    # initialize model
    model = FuseLitLenet(
        model_dir=paths["model_dir"], lr=train_params["opt.lr"], weight_decay=train_params["opt.weight_decay"]
    )

    # make datasets
    train_dataset = MNIST.dataset(paths["cache_dir"], train=True)
    validation_dataset = MNIST.dataset(paths["cache_dir"], train=False)

    # make sampler
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.label",
        num_balanced_classes=10,
        batch_size=train_params["data.batch_size"],
        balanced_class_weights=None,
    )

    # train/val dataloaders
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

    # create pl trainer
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train_params["trainer.num_epochs"],
        accelerator=train_params["trainer.accelerator"],
        strategy=train_params["trainer.strategy"],
        devices=train_params["trainer.num_devices"],
        auto_select_gpus=True,
    )

    # train model
    pl_trainer.fit(model, train_dataloader, validation_dataloader, ckpt_path=train_params["trainer.ckpt_path"])


if __name__ == "__main__":
    run_train(paths=PATHS, train_params=TRAIN_PARAMS)
