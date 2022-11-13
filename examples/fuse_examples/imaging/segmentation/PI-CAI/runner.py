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

from collections import OrderedDict
import os
import sys
import copy
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds

from fuse.utils.utils_debug import FuseDebug
from fuse.utils.gpu import choose_and_enable_multiple_gpus

import logging

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils import NDict
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.split import dataset_balanced_division_to_folds
from fuse.dl.models import ModelMultiHead
from fuse.dl.models.heads.head_global_pooling_classifier import HeadGlobalPoolingClassifier
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.losses.segmentation.loss_dice import DiceLoss #DiceBCELoss

from unet import UNet
from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC, MetricAccuracy
from fuse.eval.metrics.detection.metrics_detection_common import MetricDetectionPICAI
from fuseimg.datasets.picai import PICAI
from fuse.dl.models.backbones.backbone_resnet_3d import BackboneResnet3D
from fuse.dl.models import ModelMultiHead
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.models.heads.heads_3D import Head3D
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe
from pytorch_lightning import Trainer
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.utils.file_io.file_io import create_dir, load_pickle, save_dataframe
from fuse.eval.evaluator import EvaluatorDefault
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# assert (
#     "PICAI_DATA_PATH" in os.environ
# ), "Expecting environment variable CMMD_DATA_PATH to be set. Follow the instruction in example README file to download and set the path to the data"
##########################################
# Debug modes
##########################################
mode = "default"  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

def print_struct(d, level=0):
    if type(d) == dict or type(d) == NDict:
        keys = d.keys()
        level += 1
        for key in keys:
            print('---' * level, key)
            print_struct(d[key], level)
    else:
        if hasattr(d, 'shape'):
            print(d.shape)

def pre_proc_batch(in_batch): # [N, C, D, H, W]
    out_batch = []
    for batch in in_batch:
        if len(batch.shape) > 4:
            # reshape
            shape = batch.shape
            batch = batch.transpose(1,2)
            batch = batch.view(-1, shape[1], shape[-2], shape[-1])
        out_batch.append(batch)

    return out_batch # [N * D, C, H, W]


def post_proc_batch(out_model): # [N * D, C, H, W]
    return torch.unsqueeze(out_model,dim=0).transpose(1,2) # [N, C, D, H, W]

def create_model(train: NDict, paths: NDict) -> torch.nn.Module:
    """
    creates the model
    See HeadGlobalPoolingClassifier for details
    """
    if train["target"] == "classification":
        num_classes = 2
        gt_label = "data.gt.classification"
        skip_keys = ["data.gt.subtype"]
        class_names = ["Benign", "Malignant"]
        model = ModelMultiHead(
            conv_inputs=(('data.input.img_t2w', 1),),
            backbone=BackboneResnet3D(in_channels=1),
            heads=[
                Head3D(head_name='head_0',
                                conv_inputs=[("model.backbone_features", 512)],
                                #  dropout_rate=train_params['imaging_dropout'],
                                #  append_dropout_rate=train_params['clinical_dropout'],
                                #  fused_dropout_rate=train_params['fused_dropout'],
                                num_outputs=num_classes,
                                #  append_features=[("data.input.clinical", 8)],
                                #  append_layers_description=(256,128),
                                ),
            ])
    # elif train["target"] == "subtype":
    #     num_classes = 4
    #     gt_label = "data.gt.subtype"
    #     skip_keys = ["data.gt.classification"]
    #     class_names = ["Luminal A", "Luminal B", "HER2-enriched", "triple negative"]
    elif train['target'] == 'segmentation':
        num_classes = 2
        gt_label = "data.gt.seg"
        skip_keys = ["data.gt.subtype"]
        class_names = ["Benign", "Malignant"]
        torch_model = UNet(n_channels=1, n_classes=1, bilinear=False)

        model = ModelWrapSeqToDict(model=torch_model,
                                model_inputs=['data.input.img_t2w'],
                                model_outputs=['model.logits.segmentation'],
                                pre_forward_processing_function=pre_proc_batch,
                                post_forward_processing_function=post_proc_batch
                                )
    else:
        raise ("unsuported target!!")

    return model, num_classes, gt_label, skip_keys, class_names


#################################
# Train Template
#################################
def run_train(paths: NDict, train: NDict) -> torch.nn.Module:
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

    model, num_classes, gt_label, skip_keys, class_names = create_model(train, paths)
    lgr.info("Model: Done", {"attrs": "bold"})

    lgr.info("\nFuse Train", {"attrs": ["bold", "underline"]})

    lgr.info(f'model_dir={paths["model_dir"]}', {"color": "magenta"})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {"color": "magenta"})

    #### Train Data
    # split to folds randomly - temp
    dataset_all = PICAI.dataset(
        paths = paths,
        train_cfg = train,
        reset_cache=False,
        train=True,
    )
    folds = dataset_balanced_division_to_folds(
        dataset=dataset_all,
        output_split_filename=os.path.join(paths["data_misc_dir"], paths["data_split_filename"]),
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
        paths = paths,
        train_cfg = train,
        reset_cache=False,
        sample_ids=train_sample_ids,
        train=True,
    )

    validation_dataset = PICAI.dataset(
        paths = paths,
        train_cfg = train,
        reset_cache=False,
        sample_ids=validation_sample_ids
    )

    ## Create sampler
    lgr.info("- Create sampler:")
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name= "data.gt.classification", #gt_label, TODO - make diff label for balance-sampler
        num_balanced_classes=num_classes,
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
        collate_fn=CollateDefault(skip_keys=skip_keys,special_handlers_keys={"data.input.img":CollateDefault.pad_all_tensors_to_same_size}),
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
        batch_size=1, # TODO - set a validation batch_size parameter instead of - train["batch_size"],
        num_workers=train["num_workers"],
        collate_fn=CollateDefault(skip_keys=skip_keys),
    )
    lgr.info("Validation Data: Done", {"attrs": "bold"})

    # for x in train_dataloader:
    #     x.print_tree()
    #     out = model(x)
    #     out.print_tree()
    #     break

    # ====================================================================================
    #  Loss
    # ====================================================================================
    # TODO - add a classification loss - add head to the bottom of the unet
    losses = {
        # "cls_loss": LossDefault(pred="model.logits.head_0", target=gt_label, callable=F.cross_entropy, weight=1.0)
        'dice_loss': DiceLoss(pred_name='model.logits.segmentation', 
                                 target_name='data.gt.seg')
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    train_metrics =OrderedDict(
        # [
        #     ("picai_metric", MetricDetectionPICAI(pred='model.logits.segmentation', 
        #                          target='data.gt.seg')),  # will apply argmax
        # ]
    )

    validation_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
    best_epoch_source = dict(
        monitor="validation.losses.total_loss",  #metrics.auc.macro_avg",
        mode="min",
    )

    # =====================================================================================
    #  Manager - Train
    #  Create a manager, training objects and run a training process.
    # =====================================================================================
    lgr.info("Train:", {"attrs": "bold"})

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=train["learning_rate"], weight_decay=train["weight_decay"])
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
        auto_select_gpus=True,
    )

    # train from scratch
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader, ckpt_path=train["trainer"]["ckpt_path"])
    lgr.info("Train: Done", {"attrs": "bold"})


######################################
# Inference Template
######################################
def run_infer(train: NDict, paths: NDict, infer: NDict):
    create_dir(paths["inference_dir"])
    #### Logger
    fuse_logger_start(output_path=paths["inference_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Inference", {"attrs": ["bold", "underline"]})
    infer_file = os.path.join(paths["inference_dir"], infer["infer_filename"])
    checkpoint_file = os.path.join(paths["model_dir"], infer["checkpoint"])
    lgr.info(f"infer_filename={checkpoint_file}", {"color": "magenta"})

    lgr.info("Model:", {"attrs": "bold"})

    model, num_classes, gt_label, skip_keys, class_names = create_model(train, paths)
    lgr.info("Model: Done", {"attrs": "bold"})
    ## Data
    folds = load_pickle(
        os.path.join(paths["data_misc_dir"], paths["data_split_filename"])
    )  # assume exists and created in train func

    infer_sample_ids = []
    for fold in infer["infer_folds"]:
        infer_sample_ids += folds[fold]

    test_dataset = PICAI.dataset(
        paths["data_dir"],
        paths["data_misc_dir"],
        infer["target"],
        paths["cache_dir"],
        sample_ids=infer_sample_ids,
        train=False,
    )
    ## Create dataloader
    infer_dataloader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        drop_last=False,
        collate_fn=CollateDefault(),
        num_workers=infer["num_workers"],
    )
    # load python lightning module
    pl_module = LightningModuleDefault.load_from_checkpoint(
        checkpoint_file, model_dir=paths["model_dir"], model=model, map_location="cpu", strict=True
    )
    # set the prediction keys to extract (the ones used be the evaluation function).
    pl_module.set_predictions_keys(
        ["model.output.head_0", "data.gt.classification"]
    )  # which keys to extract and dump into file
    lgr.info("Test Data: Done", {"attrs": "bold"})
    # create lightining trainer.
    pl_trainer = Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train["trainer"]["num_epochs"],
        accelerator=train["trainer"]["accelerator"],
        devices=train["trainer"]["devices"],
        num_sanity_val_steps=-1,
        auto_select_gpus=True,
    )
    # create a trainer instance
    predictions = pl_trainer.predict(pl_module, infer_dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)


######################################
# Analyze Template
######################################
def run_eval(paths: NDict, infer: NDict):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Eval", {"attrs": ["bold", "underline"]})

    # metrics
    metrics = OrderedDict(
        [
            ("op", MetricApplyThresholds(pred="model.output.head_0")),  # will apply argmax
            ("auc", MetricAUCROC(pred="model.output.head_0", target="data.gt.classification")),
            ("accuracy", MetricAccuracy(pred="results:metrics.op.cls_pred", target="data.gt.classification")),
        ]
    )

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(
        ids=None,
        data=os.path.join(paths["inference_dir"], infer["infer_filename"]),
        metrics=metrics,
        output_dir=paths["eval_dir"],
    )

    return results


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = NDict(OmegaConf.to_object(cfg))
    print(cfg)
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    choose_and_enable_multiple_gpus(cfg["train.trainer.devices"], force_gpus=force_gpus)

    # Path to the stored dataset location
    # dataset should be download from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508
    # download requires NBIA data retriever https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
    # put on the following in the main folder  -
    # 1. CMMD_clinicaldata_revision.csv which is a converted version of CMMD_clinicaldata_revision.xlsx
    # 2. folder named CMMD which is the downloaded data folder

    # train
    if "train" in cfg["run.running_modes"]:
        run_train(cfg["paths"], cfg["train"])

    # infer
    if "infer" in cfg["run.running_modes"]:
        run_infer(cfg["train"], cfg["paths"], cfg["infer"])
    #
    # analyze
    if "eval" in cfg["run.running_modes"]:
        run_eval(cfg["paths"], cfg["infer"])


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=working_dir")
    main()
