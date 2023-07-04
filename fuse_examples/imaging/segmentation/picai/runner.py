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
import copy
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from fuse.utils.utils_debug import FuseDebug
from fuse.utils.gpu import choose_and_enable_multiple_gpus

import logging
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils import NDict
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.split import dataset_balanced_division_to_folds
from fuse.dl.losses.loss_default import LossDefault

# from report_guided_annotation import extract_lesion_candidates
from fuse_examples.imaging.segmentation.picai.unet import UNet
from fuse.eval.metrics.segmentation.metrics_segmentation_common import MetricDice
from collections import OrderedDict
from fuse.dl.losses.segmentation.loss_dice import BinaryDiceLoss

# from fuse.eval.metrics.detection.metrics_detection_common import MetricDetectionPICAI
from fuseimg.datasets.picai import PICAI
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe
from pytorch_lightning import Trainer
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.utils.file_io.file_io import create_dir, load_pickle, save_dataframe
from fuse.eval.evaluator import EvaluatorDefault

# from picai_baseline.unet.training_setup.default_hyperparam import \
#     get_default_hyperparams
# from picai_baseline.unet.training_setup.neural_network_selector import \
#     neural_network_for_run
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# from monai.losses import DiceFocalLoss

# assert (
#     "PICAI_DATA_PATH" in os.environ
# ), "Expecting environment variable PICAI_DATA_PATH to be set. Follow the instruction in example README file to download and set the path to the data"
##########################################
# Debug modes
##########################################
mode = "default"  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)


def create_model(unet_kwargs: dict) -> torch.nn.Module:
    """
    creates the model
    """
    model = UNet(
        input_name="data.input.img_t2w",
        seg_name="model.seg",
        unet_kwargs=unet_kwargs,
    )
    return model


#################################
# Train Template
#################################
def run_train(paths: NDict, train: NDict) -> None:
    # ==============================================================================
    # Logger
    # ==============================================================================
    # fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")

    # Download data
    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info("Model:", {"attrs": "bold"})

    model = create_model(train["unet_kwargs"])
    lgr.info("Model: Done", {"attrs": "bold"})

    lgr.info("\nFuse Train", {"attrs": ["bold", "underline"]})

    lgr.info(f'model_dir={paths["model_dir"]}', {"color": "magenta"})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {"color": "magenta"})

    #### Train Data
    # split to folds randomly - temp
    dataset_all = PICAI.dataset(
        paths=paths,
        cfg=train,
        reset_cache=False,
        train=True,
        run_sample=train["run_sample"],
    )
    folds = dataset_balanced_division_to_folds(
        dataset=dataset_all,
        output_split_filename=os.path.join(
            paths["data_misc_dir"], paths["data_split_filename"]
        ),
        id="data.patientID",
        keys_to_balance=["data.gt.classification"],
        nfolds=train["num_folds"],
        workers=train["num_workers"],
    )

    train_sample_ids = []
    for fold in train["train_folds"]:
        train_sample_ids += folds[fold]
    validation_sample_ids = []
    for fold in train["validation_folds"]:
        validation_sample_ids += folds[fold]

    train_dataset = PICAI.dataset(
        paths=paths,
        cfg=train,
        reset_cache=False,
        sample_ids=train_sample_ids,
        train=True,
    )

    validation_dataset = PICAI.dataset(
        paths=paths, cfg=train, reset_cache=False, sample_ids=validation_sample_ids
    )

    ## Create sampler
    lgr.info("- Create sampler:")
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.gt.classification",
        num_balanced_classes=2,
        batch_size=train["batch_size"],
        mode="approx",
        workers=train["num_workers"],
        balanced_class_weights=None,
    )

    lgr.info("- Create sampler: Done")

    ## Create dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=sampler,
        collate_fn=CollateDefault(
            special_handlers_keys={
                "data.input.img": CollateDefault.pad_all_tensors_to_same_size
            },
            raise_error_key_missing=False,
        ),
        num_workers=train["num_workers"],
    )
    lgr.info("Train Data: Done", {"attrs": "bold"})

    #### Validation data
    lgr.info("Validation Data:", {"attrs": "bold"})

    ## Create dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=train["batch_size"],
        num_workers=train["num_workers"],
        collate_fn=CollateDefault(),
    )
    lgr.info("Validation Data: Done", {"attrs": "bold"})

    # ====================================================================================
    #  Loss and metrics
    # ====================================================================================
    # TODO - add a classification loss - add head to the bottom of the unet
    losses = {}
    losses["segmentation"] = LossDefault(
        pred="model.seg",
        target="data.gt.seg",
        callable=BinaryDiceLoss(),
    )
    train_metrics = {}

    validation_metrics = copy.deepcopy(
        train_metrics
    )  # use the same metrics in validation as well

    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
    best_epoch_source = dict(
        monitor="validation.losses.total_loss",
        mode="min",
    )

    # =====================================================================================
    #  Manager - Train
    #  Create a manager, training objects and run a training process.
    # =====================================================================================
    lgr.info("Train:", {"attrs": "bold"})

    # create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train["learning_rate"],
        weight_decay=train["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    lr_sch_config = dict(scheduler=scheduler, monitor="validation.losses.total_loss")

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
    pl_trainer = Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train["trainer"]["num_epochs"],
        accelerator=train["trainer"]["accelerator"],
        devices=train["trainer"]["devices"],
        num_sanity_val_steps=-1,
    )

    # train from scratch
    pl_trainer.fit(
        pl_module,
        train_dataloader,
        validation_dataloader,
        ckpt_path=train["trainer"]["ckpt_path"],
    )
    lgr.info("Train: Done", {"attrs": "bold"})


######################################
# Inference Template
######################################
def run_infer(infer: NDict, paths: NDict, train: NDict) -> None:
    create_dir(paths["inference_dir"])
    #### Logger
    # fuse_logger_start(output_path=paths["inference_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Inference", {"attrs": ["bold", "underline"]})
    infer_file = os.path.join(paths["inference_dir"], infer["infer_filename"])
    checkpoint_file = os.path.join(paths["model_dir"], infer["checkpoint"])
    lgr.info(f"infer_filename={checkpoint_file}", {"color": "magenta"})

    lgr.info("Model:", {"attrs": "bold"})

    model = create_model(train["unet_kwargs"])
    lgr.info("Model: Done", {"attrs": "bold"})
    ## Data
    folds = load_pickle(
        os.path.join(paths["data_misc_dir"], paths["data_split_filename"])
    )  # assume exists and created in train func

    infer_sample_ids = []
    for fold in infer["infer_folds"]:
        infer_sample_ids += folds[fold]

    test_dataset = PICAI.dataset(
        paths=paths,
        cfg=train,
        sample_ids=infer_sample_ids,
        reset_cache=False,
    )

    ## Create dataloader
    infer_dataloader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=train["batch_size"],
        collate_fn=CollateDefault(),
        num_workers=infer["num_workers"],
    )
    # load python lightning module
    pl_module = LightningModuleDefault.load_from_checkpoint(
        checkpoint_file,
        model_dir=paths["model_dir"],
        model=model,
        map_location="cpu",
        strict=True,
    )
    # set the prediction keys to extract (the ones used be the evaluation function).
    pl_module.set_predictions_keys(
        ["model.seg", "data.gt.seg"]
    )  # which keys to extract and dump into file
    lgr.info("Test Data: Done", {"attrs": "bold"})
    # create lightining trainer.
    pl_trainer = Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train["trainer"]["num_epochs"],
        accelerator=train["trainer"]["accelerator"],
        devices=train["trainer"]["devices"],
        num_sanity_val_steps=-1,
    )
    # create a trainer instance
    predictions = pl_trainer.predict(
        pl_module, infer_dataloader, return_predictions=True
    )

    with open(Path(infer_file).parents[0] / "infer.pickle", "wb") as f:
        pickle.dump(predictions, f)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)


######################################
# Analyze Template
######################################
def run_eval(paths: NDict, infer: NDict) -> None:
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Eval", {"attrs": ["bold", "underline"]})

    # create evaluator
    evaluator = EvaluatorDefault()
    metrics = OrderedDict(
        [
            ("dice", MetricDice(pred="model.seg", target="data.gt.seg")),
        ]
    )
    # define iterator

    def data_iter() -> NDict:
        data_file = os.path.join(paths["inference_dir"], "infer.pickle")
        data = pd.read_pickle(data_file)
        for fold in data:
            for i, sample in enumerate(fold["id"]):
                sample_dict = {}
                sample_dict["id"] = sample
                sample_dict["model.seg"] = np.expand_dims(
                    fold["model.seg"][i][1], axis=0
                )
                sample_dict["model.seg"] = np.where(
                    sample_dict["model.seg"] > 0.7, 1, 0
                )
                sample_dict["data.gt.seg"] = fold["data.gt.seg"][i]
                yield sample_dict

    # run
    results = evaluator.eval(
        ids=None,
        data=data_iter(),
        batch_size=60,
        metrics=metrics,
        output_dir=paths["eval_dir"],
    )
    print(results)

    return results


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = NDict(OmegaConf.to_object(cfg))
    print(cfg)
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    choose_and_enable_multiple_gpus(cfg["train.trainer.devices"], force_gpus=force_gpus)

    # train
    if "train" in cfg["run.running_modes"]:
        run_train(NDict(cfg["paths"]), NDict(cfg["train"]))

    # infer
    if "infer" in cfg["run.running_modes"]:
        run_infer(NDict(cfg["infer"]), NDict(cfg["paths"]), NDict(cfg["train"]))

    # analyze
    if "eval" in cfg["run.running_modes"]:
        run_eval(NDict(cfg["paths"]), NDict(cfg["infer"]))


if __name__ == "__main__":
    main()
