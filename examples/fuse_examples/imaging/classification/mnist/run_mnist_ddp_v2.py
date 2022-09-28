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

MNIST classifier implementation that demonstrate end to end training, inference and evaluation using FuseMedML with DDP strategy
"""

import copy
import logging
import os
from typing import OrderedDict, Tuple
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe

import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from fuse.eval.evaluator import EvaluatorDefault
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve

from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.lightning.pl_module import LightningModuleDefault

from fuse.utils.utils_debug import FuseDebug
from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import create_dir, save_dataframe
import fuse.utils.gpu as GPU

from fuse_examples.imaging.classification.mnist.lenet import LeNet
from fuse_examples.imaging.classification.mnist.mnist_data_module import MNISTDataModule

"""
So you want to use distributed data parallel (DDP)[1] strategy to increase your batch size or boost your training?
FuseMedML supports DDP strategy based on PyTorch-Lightning [2].

The following example shows how to use DDP with FuseMedML on the famous MNIST dataset.
NOTE that if you want to use FuseMedML's custom batch sampler 'BatchSamplerDefault' sampler, you shall implement a datamodule similar to 'MNISTDataModule'. (relevant for PyTorch-Lightning 1.7.6)


@rank_zero_only
A PyTorch-Lightning decorator that makes sure the function runs only in the main proccess (where the RANK is 0).
Helpful to avoid printing / logging multiple time.

[1] PyTorch: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
[2] PyTorch-Lightning: https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_intermediate.html
"""
###########################################################################################################
# Fuse
###########################################################################################################

##########################################
# Debug modes
##########################################
mode = "default"
debug = FuseDebug(mode)

##########################################
# Output Paths
##########################################
ROOT = "_examples/mnist_ddp"
model_dir = os.path.join(ROOT, "model_dir")
PATHS = {
    "model_dir": model_dir,
    "cache_dir": os.path.join(ROOT, "cache_dir"),
    "inference_dir": os.path.join(model_dir, "infer_dir"),
    "eval_dir": os.path.join(model_dir, "eval_dir"),
    "data_split_filename": os.path.join(model_dir, "mnist_split.pkl"),
}

NUM_GPUS = 2  # Multiple GPU training
WORKERS = 10
##########################################
# Train Common Params
##########################################
TRAIN_COMMON_PARAMS = {}

# ============
# Data
# ============
TRAIN_COMMON_PARAMS["data.batch_size"] = 40
TRAIN_COMMON_PARAMS["data.num_workers"] = WORKERS

# ===============
# PL Trainer
# ===============
TRAIN_COMMON_PARAMS["trainer.num_epochs"] = 1
TRAIN_COMMON_PARAMS["trainer.num_devices"] = NUM_GPUS
TRAIN_COMMON_PARAMS["trainer.accelerator"] = "gpu"
TRAIN_COMMON_PARAMS["trainer.strategy"] = "ddp"

# ===============
# Optimizer
# ===============
TRAIN_COMMON_PARAMS["opt.lr"] = 1e-4
TRAIN_COMMON_PARAMS["opt.weight_decay"] = 0.001


def perform_softmax(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cls_preds = F.softmax(logits, dim=1)
    return logits, cls_preds


def create_model() -> torch.nn.Module:
    torch_model = LeNet()
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
def run_train(paths: dict, train_params: dict):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)
    print("Fuse Train")

    # ==============================================================================
    # Data
    # ==============================================================================

    print("- Create DataModule:")

    datamodule = MNISTDataModule(
        cache_dir=paths["cache_dir"],
        batch_size=train_params["data.batch_size"],
        num_workers=train_params["data.num_workers"],
        train_folds=[1, 2, 3, 4],
        validation_folds=[5],
        split_filename = paths["data_split_filename"],
    )

    print("- Create DataModule: Done")

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

    # optimizier and lr sch
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
    )

    # train
    pl_trainer.fit(pl_module, datamodule=datamodule)
    print("Train: Done")


######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS["infer_filename"] = "infer_file.gz"
INFER_COMMON_PARAMS["checkpoint"] = "best_epoch.ckpt"
INFER_COMMON_PARAMS["trainer.num_devices"] = 1
INFER_COMMON_PARAMS["trainer.accelerator"] = "gpu"

######################################
# Inference Template
######################################


def run_infer(paths: dict, infer_common_params: dict):
    create_dir(paths["inference_dir"])
    infer_file_path = os.path.join(paths["inference_dir"], infer_common_params["infer_filename"])
    checkpoint_file = os.path.join(paths["model_dir"], infer_common_params["checkpoint"])

    #### Logger
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)

    print("Fuse Inference")
    print(f"infer_file_path={infer_file_path}")

    ## Data
    datamodule = MNISTDataModule(cache_dir=paths["cache_dir"], num_workers=2, batch_size=10)

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
        accelerator=infer_common_params["trainer.accelerator"],
        devices=infer_common_params["trainer.num_devices"],
        auto_select_gpus=True,
    )
    predictions = pl_trainer.predict(pl_module, datamodule=datamodule, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file_path)


######################################
# Analyze Common Params
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS["infer_filename"] = INFER_COMMON_PARAMS["infer_filename"]


######################################
# Eval Template
######################################
@rank_zero_only
def run_eval(paths: dict, eval_common_params: dict):
    create_dir(paths["eval_dir"])
    infer_file_path = os.path.join(paths["inference_dir"], eval_common_params["infer_filename"])

    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)

    print("Fuse Eval")

    # metrics
    class_names = [str(i) for i in range(10)]

    metrics = OrderedDict(
        [
            ("operation_point", MetricApplyThresholds(pred="model.output.classification")),
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
    results = evaluator.eval(ids=None, data=infer_file_path, metrics=metrics, output_dir=paths["eval_dir"])

    return results


######################################
# Run
######################################
if __name__ == "__main__":

    GPU.choose_and_enable_multiple_gpus(NUM_GPUS)
    RUNNING_MODES = ["train", "infer", "eval"]

    # train
    if "train" in RUNNING_MODES:
        run_train(paths=PATHS, train_params=TRAIN_COMMON_PARAMS)

    # infer
    if "infer" in RUNNING_MODES:
        run_infer(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    # eval - runs only on the main process (zero local rank)
    if "eval" in RUNNING_MODES:
        run_eval(paths=PATHS, eval_common_params=EVAL_COMMON_PARAMS)
