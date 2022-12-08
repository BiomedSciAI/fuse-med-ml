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

MNIST classifier implementation that demonstrate end to end training, inference and evaluation using FuseMedML
"""

import copy
import logging
import os
from typing import OrderedDict, Tuple
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from fuse.eval.evaluator import EvaluatorDefault
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve

from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault

from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.lightning.pl_module import LightningModuleDefault

from fuse.utils.utils_debug import FuseDebug
from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import create_dir, save_dataframe
import fuse.utils.gpu as GPU

from fuseimg.datasets.mnist import MNIST
from fuse.data.datasets.dataset_default import DatasetDefault
from examples.fuse_examples.imaging.classification.mnist import lenet

###########################################################################################################
# Fuse
###########################################################################################################

##########################################
# Debug modes
##########################################
mode = "default"  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################
# Output Paths
##########################################
ROOT = "_examples/mnist"  # TODO: fill path here
model_dir = os.path.join(ROOT, "model_dir")
PATHS = {
    "model_dir": model_dir,
    "cache_dir": os.path.join(ROOT, "cache_dir"),
    "inference_dir": os.path.join(model_dir, "infer_dir"),
    "eval_dir": os.path.join(model_dir, "eval_dir"),
}

NUM_GPUS = 1
##########################################
# Train Common Params
##########################################
TRAIN_COMMON_PARAMS = {}

# ============
# Data
# ============
TRAIN_COMMON_PARAMS["data.batch_size"] = 100
TRAIN_COMMON_PARAMS["data.train_num_workers"] = 8
TRAIN_COMMON_PARAMS["data.validation_num_workers"] = 8

# ===============
# PL Trainer
# ===============
TRAIN_COMMON_PARAMS["trainer.num_epochs"] = 2
TRAIN_COMMON_PARAMS["trainer.num_devices"] = NUM_GPUS
TRAIN_COMMON_PARAMS["trainer.accelerator"] = "gpu"
# use "dp" strategy temp when working with multiple GPUS - workaround for pytorch lightning issue: https://github.com/Lightning-AI/lightning/issues/11807
TRAIN_COMMON_PARAMS["trainer.strategy"] = "dp" if TRAIN_COMMON_PARAMS["trainer.num_devices"] > 1 else None
TRAIN_COMMON_PARAMS["trainer.ckpt_path"] = None  # path to the checkpoint you wish continue the training from

# ===============
# Optimizer
# ===============
TRAIN_COMMON_PARAMS["opt.lr"] = 1e-4
TRAIN_COMMON_PARAMS["opt.weight_decay"] = 0.001


def perform_softmax(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cls_preds = F.softmax(logits, dim=1)
    return logits, cls_preds


def create_model() -> torch.nn.Module:
    torch_model = lenet.LeNet()
    # wrap basic torch model to automatically read inputs from batch_dict and save its outputs to batch_dict
    model = ModelWrapSeqToDict(
        model=torch_model,
        model_inputs=["data.image"],
        post_forward_processing_function=perform_softmax,
        model_outputs=["model.logits.classification", "model.output.classification"],
    )
    return model


#################################
# Train Template
#################################
def run_train(train_dataset: DatasetDefault, validation_dataset: DatasetDefault, paths: dict, train_params: dict):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)
    print("Fuse Train")

    # ==============================================================================
    # Data
    # ==============================================================================
    print("Data - training set:")
    print("- Create sampler:")
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.label",
        num_balanced_classes=10,
        batch_size=train_params["data.batch_size"],
        balanced_class_weights=None,
    )
    print("- Create sampler: Done")

    # Create dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=train_params["data.train_num_workers"],
    )
    print("Data - training set: Done")

    ## Validation data
    print("Data - validation set:")

    # dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=train_params["data.batch_size"],
        collate_fn=CollateDefault(),
        num_workers=train_params["data.validation_num_workers"],
    )
    print("Data - validation set: Done")

    # ====================================================================================
    # Model
    # ====================================================================================
    model = create_model()

    # ====================================================================================
    # Losses
    # ====================================================================================
    losses = {
        "cls_loss": LossDefault(
            pred="model.logits.classification", target="data.label", callable=F.cross_entropy, weight=1.0
        ),
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    train_metrics = OrderedDict(
        [
            ("operation_point", MetricApplyThresholds(pred="model.output.classification")),  # will apply argmax
            ("accuracy", MetricAccuracy(pred="results:metrics.operation_point.cls_pred", target="data.label")),
        ]
    )

    validation_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
    best_epoch_source = dict(
        monitor="validation.metrics.accuracy",
        mode="max",
    )

    # ====================================================================================
    # Training components
    # ====================================================================================
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=train_params["opt.lr"], weight_decay=train_params["opt.weight_decay"])

    # create learning scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")

    # optimizer and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    # ====================================================================================
    # Train
    # ====================================================================================
    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=paths["model_dir"],
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )

    # create lightining trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train_params["trainer.num_epochs"],
        accelerator=train_params["trainer.accelerator"],
        strategy=train_params["trainer.strategy"],
        devices=train_params["trainer.num_devices"],
        auto_select_gpus=True,
    )

    # train
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader, ckpt_path=train_params["trainer.ckpt_path"])
    print("Train: Done")


######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS["infer_filename"] = "infer_file.gz"
INFER_COMMON_PARAMS["checkpoint"] = "best_epoch.ckpt"
INFER_COMMON_PARAMS["trainer.num_devices"] = 1
INFER_COMMON_PARAMS["trainer.accelerator"] = "gpu"
INFER_COMMON_PARAMS["trainer.strategy"] = None

######################################
# Inference Template
######################################


def run_infer(dataset: DatasetDefault, paths: dict, infer_params: dict):
    create_dir(paths["inference_dir"])
    infer_file = os.path.join(paths["inference_dir"], infer_params["infer_filename"])
    checkpoint_file = os.path.join(paths["model_dir"], infer_params["checkpoint"])
    #### Logger
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)

    print("Fuse Inference")
    print(f"infer_filename={infer_file}")

    ## Data
    # dataloader
    dataloader = DataLoader(dataset=dataset, collate_fn=CollateDefault(), batch_size=2, num_workers=2)

    # load pytorch lightning module
    model = create_model()
    pl_module = LightningModuleDefault.load_from_checkpoint(
        checkpoint_file, model_dir=paths["model_dir"], model=model, map_location="cpu", strict=True
    )
    # set the prediction keys to extract (the ones used be the evaluation function).
    pl_module.set_predictions_keys(
        ["model.output.classification", "data.label"]
    )  # which keys to extract and dump into file

    print("Model: Done")
    # create a trainer instance
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        accelerator=infer_params["trainer.accelerator"],
        devices=infer_params["trainer.num_devices"],
        strategy=infer_params["trainer.strategy"],
        auto_select_gpus=True,
    )
    predictions = pl_trainer.predict(pl_module, dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)


######################################
# Analyze Common Params
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS["infer_filename"] = INFER_COMMON_PARAMS["infer_filename"]


######################################
# Eval Template
######################################
def run_eval(paths: dict, eval_params: dict):
    create_dir(paths["eval_dir"])
    infer_file = os.path.join(paths["inference_dir"], eval_params["infer_filename"])
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)

    print("Fuse Eval")

    # metrics
    class_names = [str(i) for i in range(10)]

    metrics = OrderedDict(
        [
            ("operation_point", MetricApplyThresholds(pred="model.output.classification")),  # will apply argmax
            ("accuracy", MetricAccuracy(pred="results:metrics.operation_point.cls_pred", target="data.label")),
            (
                "roc",
                MetricROCCurve(
                    pred="model.output.classification",
                    target="data.label",
                    class_names=class_names,
                    output_filename=os.path.join(paths["eval_dir"], "roc_curve.png"),
                ),
            ),
            ("auc", MetricAUCROC(pred="model.output.classification", target="data.label", class_names=class_names)),
        ]
    )

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(ids=None, data=infer_file, metrics=metrics, output_dir=paths["eval_dir"])

    return results


######################################
# Run
######################################
if __name__ == "__main__":
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    RUNNING_MODES = ["train", "infer", "eval"]  # Options: 'train', 'infer', 'eval'
    train_dataset = MNIST.dataset(PATHS["cache_dir"], train=True)
    validation_dataset = MNIST.dataset(PATHS["cache_dir"], train=False)
    # train
    if "train" in RUNNING_MODES:
        run_train(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            paths=PATHS,
            train_params=TRAIN_COMMON_PARAMS,
        )

    # infer
    if "infer" in RUNNING_MODES:
        run_infer(dataset=validation_dataset, paths=PATHS, infer_params=INFER_COMMON_PARAMS)

    # eval
    if "eval" in RUNNING_MODES:
        run_eval(paths=PATHS, eval_params=EVAL_COMMON_PARAMS)
