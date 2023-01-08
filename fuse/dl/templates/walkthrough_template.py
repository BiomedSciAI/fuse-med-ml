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
from typing import OrderedDict
import logging

import torch
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from fuse.utils.utils_debug import FuseDebug
import fuse.utils.gpu as GPU
from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import create_dir, save_dataframe

from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault

from fuse.dl.models import ModelMultiHead
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe

from fuse.eval.evaluator import EvaluatorDefault

###########################################################################################################
# Fuse
# Example of a training template:
#
# Training template contains 5 fundamental building blocks:
# (1) Data - detailed README file can be found at [fuse/data](../../fuse/data)
# (2) Model
# (3) Losses
# (4) Metrics and Evaluation - detailed README file can be found at [fuse/eval](../../fuse/eval)
# (5) Train - using PyTorch Lightning
#
# The template will create each of those building blocks and will eventually run pl_trainer.fit()
#
# Terminology:
# 1. sample-id -
#    A unique identifier of a sample. Each sample in the dataset must have an id that uniquely identifies it.
#    Examples of sample ids:
#    * path to the image file
#    * Tuple of (provider_id, patient_id, image_id)
#    * Running index
# 2. sample_dict -
#    Represents a single sample and contains all relevant information about the sample.
#    No specific structure of this dictionary is required, but a useful pattern is to split it into sections (keys that define a "namespace" ): such as "data", "model",  etc.
#    NDict (fuse/utils/ndict.py) class is used instead of python standard dictionary in order to allow easy "." separated access. For example:
#    `sample_dict[“data.input.img”]` is the equivalent of `sample_dict["data"]["input"]["img"]`
#    Another recommended convention is to include suffix specifying the type of the value ("img", "seg", "bbox")
# 3. Fuse base classes - *Base -
#    Abstract classes of the object forming together Fuse framework .
# 4. Fuse default classes - *Default -
#    A default generic implementation of the equivalent base class.
#    Those generic implementation will be useful for most common use cases.
#    Alternative implementations could be implemented for the special cases.
###########################################################################################################

##########################################
# Debug modes
##########################################
mode = "default"  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################
# Output Paths
##########################################
ROOT = None  # TODO: fill in a path to model dir
model_dir = os.path.join(ROOT, "model_dir")
PATHS = {
    "model_dir": model_dir,
    "force_reset_model_dir": False,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
    "cache_dir": os.path.join(ROOT, "cache_dir"),
    "inference_dir": os.path.join(model_dir, "infer"),
    "eval_dir": os.path.join(model_dir, "eval"),
    "data_split_filename": os.path.join(ROOT, "split.pkl"),
}

##########################################
# Train Common Params
##########################################
TRAIN_COMMON_PARAMS = {}
# ============
# Data
# ============
TRAIN_COMMON_PARAMS["data.batch_size"] = 2
TRAIN_COMMON_PARAMS["data.train_num_workers"] = 8
TRAIN_COMMON_PARAMS["data.validation_num_workers"] = 8
TRAIN_COMMON_PARAMS["data.cache_num_workers"] = 10

# ===============
# PL Trainer
# ===============
TRAIN_COMMON_PARAMS["trainer.num_epochs"] = 100
TRAIN_COMMON_PARAMS["trainer.num_devices"] = 1
TRAIN_COMMON_PARAMS["trainer.accelerator"] = "gpu"
TRAIN_COMMON_PARAMS["trainer.ckpt_path"] = None

# ===============
# Optimizer
# ===============
TRAIN_COMMON_PARAMS["opt.lr"] = 1e-4
TRAIN_COMMON_PARAMS["opt.weight_decay"] = 1e-3

# ===================================================================================================================
# Model
#   Build a model (torch.nn.Module) using generic Fuse components:
#   1. ModelMultiHead - generic component supporting single backbone with multiple heads
#   2. Backbone - simple backbone model
#   3. Head* - generic head implementations
#   The model outputs will be aggregated in batch_dict['model.*']
#   Each head output will be aggregated in batch_dict['model.<head name>.*']
#
#   Additional implemented models:
#   * ModelWrapper - allow to use single standard PyTorch module as is - see (examples/fuse_examples/imaging/classification/mnist/runner.py)[../../examples/fuse_examples/imaging/classification/mnist/runner.py]
#   * ModelEnsemble - runs several sub-modules sequentially
#   * ModelMultistream - convolutional neural network with multiple processing streams and multiple heads
# ===================================================================================================================


def create_model() -> torch.nn.Module:
    # TODO - define / create a model
    model = ModelMultiHead(
        conv_inputs=(("data.input.input_0.tensor", 1),),
        backbone="TODO",  # Reference: BackboneInceptionResnetV2
        heads=["TODO"],  # References: HeadGlobalPoolingClassifier, HeadDenseSegmentation
    )
    return model


#################################
# Train Template
#################################
def run_train(paths: dict, train_common_params: dict) -> None:
    # ==============================================================================
    # Logger
    #   - output log automatically to three destinations:
    #     (1) console (2) file - copy of the console (3) verbose file - used for debug
    #   - save a copy of the template file
    # ==============================================================================
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)

    print("Fuse Train")
    print(f'model_dir={paths["model_dir"]}')
    print(f'cache_dir={paths["cache_dir"]}')

    # ==============================================================================
    # Data
    #   Build dataloaders (torch.utils.data.DataLoader) for both train and validation.
    #
    #   Default dataset implementation built from a sequence of op(erator)s.
    #   Operators are the building blocks of the sample processing pipeline.
    #   Each operator gets as input the *sample_dict* as created by the previous operators and can either add/delete/modify fields in sample_dict.
    #   The operator interface is specified in OpBase class.
    #
    #   We split the pipeline into two parts - static and dynamic, which allow us to control the part out of the entire pipeline that will be cached.
    #
    #   For more details and examples, read (fuse/data/README.md)[../../fuse/data/README.md]
    #   A complete example implementation of a dataset can be bound in  (fuseimg/datasets/stoic21.py)[../../examples/fuse_examples/imaging/classification/stoic21/runner_stoic21.py] STOIC21.static_pipeline().
    # ==============================================================================

    #### Train Data

    print("Train Data:")

    ## TODO - list your sample ids:
    # Fuse TIP - splitting the sample_ids to folds can be done by fuse.data.utils.split.dataset_balanced_division_to_folds().
    #            See (examples/fuse_examples/imaging/classification/stoic21/runner_stoic21.py)[../../examples/fuse_examples/imaging/classification/stoic21/runner_stoic21.py]
    train_sample_ids = None
    validation_sample_ids = None

    ## Create data static_pipeline -
    #                                the output of this pipeline will be cached to optimize the running time and to better utilize the GPU:
    #                                See example in (fuseimg/datasets/stoic21.py)[../../examples/fuse_examples/imaging/classification/stoic21/runner_stoic21.py] STOIC21.static_pipeline().
    static_pipeline = PipelineDefault(
        "template_static",
        [
            # TODO
        ],
    )
    ## Create data dynamic_pipeline - Dynamic pipeline follows the static pipeline and continues to pre-process the sample.
    #                                 In contrast to the static pipeline, the output of the dynamic pipeline is not be cached and allows modifying the pre-precessing steps without recaching,
    #                                 The recommendation is to include in the dynamic pipeline pre-processing steps that we intend to experiment with or augmentation steps.
    train_dynamic_pipeline = PipelineDefault(
        "template_dynamic",
        [
            # TODO
        ],
    )
    validation_dynamic_pipeline = PipelineDefault(
        "template_dynamic",
        [
            # TODO
        ],
    )

    # Create dataset
    cacher = SamplesCacher(
        "template_cache",
        static_pipeline,
        [paths["cache_dir"]],
        restart_cache=False,
        workers=train_common_params["data.cache_num_workers"],
    )

    train_dataset = DatasetDefault(
        sample_ids=train_sample_ids,
        static_pipeline=static_pipeline,
        dynamic_pipeline=train_dynamic_pipeline,
        cacher=cacher,
    )

    print("- Load and cache data:")
    train_dataset.create()
    print("- Load and cache data: Done")

    ## Create batch sampler
    # Fuse TIPs:
    # 1. You don't have to balance according the classification labels, any categorical value will do.
    #    Use balanced_class_name to select the categorical value
    # 2. You don't have to equally balance between the classes.
    #    Use balanced_class_weights to specify the number of required samples in a batch per each class
    # 3. Use mode to specify probabilities rather then exact number of samples from  a class in each batch
    print("- Create sampler:")
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="TODO",
        num_balanced_classes="TODO",
        batch_size=train_common_params["data.batch_size"],
        balanced_class_weights=None,
    )

    print("- Create sampler: Done")

    ## Create dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=train_common_params["data.train_num_workers"],
    )
    print("Train Data: Done")

    #### Validation data
    print("Validation Data:")

    validation_dataset = DatasetDefault(
        sample_ids=validation_sample_ids,
        static_pipeline=static_pipeline,
        dynamic_pipeline=validation_dynamic_pipeline,
        cacher=cacher,
    )

    print("- Load and cache data:")
    validation_dataset.create()
    print("- Load and cache data: Done")

    ## Create dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=train_common_params["data.batch_size"],
        num_workers=train_common_params["data.validation_num_workers"],
        collate_fn=CollateDefault(),
    )
    print("Validation Data: Done")

    ## Create model
    print("Model:")
    model = create_model()
    print("Model: Done")

    # ==========================================================================================================================================
    #   Loss
    #   Dictionary of loss elements each element is a sub-class of LossBase
    #   The total loss will be the weighted sum of all the elements.
    #   Each element output loss will be aggregated in batch_dict['losses.<loss name>']
    #   The average batch loss per epoch will be included in epoch_result['losses.<loss name>'],
    #   and the total loss in epoch_result['losses.total_loss']
    #   The 'best_epoch_source', used to save the best model could be based on one of this losses.
    #   Available Losses:
    #   LossDefault - wraps a PyTorch loss function with an api.
    #
    # ==========================================================================================================================================
    losses = {
        # TODO add losses here (instances of LossBase)
    }

    # =========================================================================================================
    # Metrics - details can be found in (fuse/eval/README.md)[../../fuse/eval/README.md]
    #   1. Create seperately for train and validation (might be a deep copy, but not a shallow one).
    #   2. Set best_epoch_source:
    #       monitor: the metric name to track
    #       mode: either consider the "min" value to be best or the "max" value to be the best
    # =========================================================================================================
    train_metrics = OrderedDict(
        [
            # TODO add metrics here (<name>, <instance of MetricBase>)
        ]
    )
    validation_metrics = OrderedDict(
        [
            # TODO add metrics here (<name>, <instance of MetricBase>)
        ]
    )

    best_epoch_source = dict(monitor="TODO", mode="TODO")

    # =====================================================================================
    #  Train - using PyTorch Lightning
    #  Create training objects, PL module and PL trainer.
    # =====================================================================================
    print("Fuse Train:")

    # create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_common_params["opt.lr"],
        weight_decay=train_common_params["opt.weight_decay"],
    )

    # create scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")

    # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

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
        max_epochs=train_common_params["trainer.num_epochs"],
        accelerator=train_common_params["trainer.accelerator"],
        devices=train_common_params["trainer.num_devices"],
        auto_select_gpus=True,
    )

    # train
    pl_trainer.fit(
        pl_module, train_dataloader, validation_dataloader, ckpt_path=train_common_params["trainer.ckpt_path"]
    )

    print("Fuse Train: Done")


######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS["data.num_workers"] = TRAIN_COMMON_PARAMS["data.train_num_workers"]
INFER_COMMON_PARAMS["data.batch_size"] = 4
INFER_COMMON_PARAMS["infer_filename"] = os.path.join(PATHS["inference_dir"], "validation_set_infer.pickle")
INFER_COMMON_PARAMS["checkpoint"] = "best_epoch.ckpt"  # Fuse TIP: possible values are 'best', 'last' or epoch_index.

######################################
# Inference Template
######################################


def run_infer(paths: dict, infer_common_params: dict) -> None:
    create_dir(paths["inference_dir"])
    infer_file = os.path.join(paths["inference_dir"], infer_common_params["infer_filename"])
    checkpoint_file = os.path.join(paths["model_dir"], infer_common_params["checkpoint"])

    ## Logger
    fuse_logger_start(output_path=paths["inference_dir"], console_verbose_level=logging.INFO)
    print("Fuse Inference")
    print(f"infer_filename={infer_file}")

    #### create infer dataset
    infer_dataset = None  # TODO: follow the same steps to create dataset as in run_train

    ## Create dataloader
    infer_dataloader = DataLoader(
        dataset=infer_dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=infer_common_params["data.batch_size"],
        num_workers=infer_common_params["data.num_workers"],
        collate_fn=CollateDefault(),
    )

    # TODO - define / create a model
    model = ModelMultiHead(
        conv_inputs=(("data.input.input_0.tensor", 1),),
        backbone="TODO",  # Reference: BackboneInceptionResnetV2
        heads=["TODO"],  # References: HeadGlobalPoolingClassifier, HeadDenseSegmentation
    )

    # load python lightning module
    pl_module = LightningModuleDefault.load_from_checkpoint(
        checkpoint_file, model_dir=paths["model_dir"], model=model, map_location="cpu", strict=True
    )

    # set the prediction keys to extract and dump into file (the ones used be the evaluation function).
    pl_module.set_predictions_keys(
        [
            # TODO
        ]
    )

    # create a trainer instance and predict
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        accelerator=infer_common_params["trainer.accelerator"],
        devices=infer_common_params["trainer.num_devices"],
        auto_select_gpus=True,
    )
    predictions = pl_trainer.predict(pl_module, infer_dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)

    print("Fuse Inference: Done")


######################################
# Eval Template
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS["infer_filename"] = INFER_COMMON_PARAMS["infer_filename"]


def run_eval(paths: dict, eval_common_params: dict) -> None:
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    print("Fuse Eval")

    # metrics
    metrics = OrderedDict(
        [
            # TODO add metrics here (<name>, <instance of MetricBase>)
        ]
    )

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(
        ids=None,
        data=os.path.join(paths["inference_dir"], eval_common_params["infer_filename"]),
        metrics=metrics,
        output_dir=paths["eval_dir"],
    )

    print("Fuse Eval: Done")
    return results


######################################
# Run
######################################
if __name__ == "__main__":
    # allocate gpus
    NUM_GPUS = 1
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    RUNNING_MODES = ["train", "infer", "eval"]  # Options: 'train', 'infer', 'eval'

    # train
    if "train" in RUNNING_MODES:
        run_train(paths=PATHS, train_common_params=TRAIN_COMMON_PARAMS)

    # infer
    if "infer" in RUNNING_MODES:
        run_infer(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    # eval
    if "eval" in RUNNING_MODES:
        run_eval(paths=PATHS, eval_common_params=EVAL_COMMON_PARAMS)
