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
from typing import OrderedDict, Any, Tuple, Optional
from jsonargparse import CLI

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DistributedSampler
import pytorch_lightning as pl

from fuse.eval.metrics.classification.metrics_thresholding_common import (
    MetricApplyThresholds,
)
from fuse.eval.metrics.classification.metrics_classification_common import (
    MetricAccuracy,
)

from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.lightning.pl_module import LightningModuleDefault

from fuseimg.datasets.mnist import MNIST
from fuse_examples.imaging.classification.mnist import lenet

import torch


def perform_softmax(logits: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    cls_preds = F.softmax(logits, dim=1)
    return logits, cls_preds


## Model Definition #####################################################
class FuseLitLenet(LightningModuleDefault):
    def __init__(
        self, model_dir: Optional[str], save_model: bool, lr: float, weight_decay: float
    ):
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
            model_outputs=[
                "model.logits.classification",
                "model.output.classification",
            ],
        )

        # losses and metrics to track
        losses = {
            "cls_loss": LossDefault(
                pred="model.logits.classification",
                target="data.label",
                callable=F.cross_entropy,
                weight=1.0,
            ),
        }
        train_metrics = OrderedDict(
            [
                (
                    "operation_point",
                    MetricApplyThresholds(pred="model.output.classification"),
                ),  # will apply argmax
                (
                    "accuracy",
                    MetricAccuracy(
                        pred="results:metrics.operation_point.cls_pred",
                        target="data.label",
                    ),
                ),
            ]
        )
        validation_metrics = copy.deepcopy(train_metrics)  # same as train metrics

        # optimizer and learning rate scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        lr_sch_config = dict(
            scheduler=lr_scheduler, monitor="validation.losses.total_loss"
        )
        optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

        # initialize LightningModuleDefault with our module, losses, etc so that we can use the functions of LightningModuleDefault
        super().__init__(
            model_dir=model_dir,
            save_model=save_model,
            model=model,
            losses=losses,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            optimizers_and_lr_schs=optimizers_and_lr_schs,
        )


## Datamodule ###########################################################
class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        distributed: bool,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed

    def setup(self, stage: str) -> None:
        self.mnist_train = MNIST.dataset(self.data_dir, train=True)
        self.mnist_val = MNIST.dataset(self.data_dir, train=False)

    def train_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self.mnist_train) if self.distributed else None
        batch_sampler = BatchSamplerDefault(
            sampler=sampler,
            dataset=self.mnist_train,
            mode="approx",
            balanced_class_name="data.label",
            num_balanced_classes=10,
            batch_size=self.batch_size,
            balanced_class_weights=None,
        )
        train_loader = DataLoader(
            dataset=self.mnist_train,
            batch_sampler=batch_sampler,
            collate_fn=CollateDefault(),
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=CollateDefault(),
        )
        return val_loader


## Training #############################################################
def run_train(
    data_dir: str = "_examples/mnist",
    batch_size: int = 100,
    num_workers: int = 8,
    num_epochs: int = 3,
    num_devices: int = 1,
    num_nodes: int = 1,
    accelerator: str = "gpu",
    ckpt_path: Optional[str] = None,
    lr: float = 1e-4,
    weight_decay: float = 0.001,
) -> None:
    # initialize model
    model = FuseLitLenet(
        model_dir=None,
        save_model=False,
        lr=lr,
        weight_decay=weight_decay,
    )

    # initialize data module
    distributed = True if num_devices > 1 else None
    mnist_dm = MNISTDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=distributed,
    )

    # A workaround to support multiple GPUs with a custom batch_sampler for both Lightning versions
    #       see: https://lightning.ai/pages/releases/2.0.0/#sampler-replacement
    kwargs = {}
    if distributed:
        if pl.__version__[0] == "2":
            kwargs["use_distributed_sampler"] = False
        else:
            kwargs["replace_sampler_ddp"] = False

    # create pl trainer
    pl_trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=num_devices,
        num_nodes=num_nodes,
        **kwargs,
    )

    # train model
    pl_trainer.fit(model, mnist_dm)


if __name__ == "__main__":
    CLI(run_train)
