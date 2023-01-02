from collections import OrderedDict
import pathlib
from fuse.utils.utils_logger import fuse_logger_start
import os
import yaml

import pandas as pd
from fuse.dl.models import ModelMultiHead
from fuse.dl.models.backbones.backbone_resnet_3d import BackboneResnet3D
# from fuse.dl.models.heads.heads_3D import Head3D
from fuse.dl.models.heads.heads_3D import Head3DClassifier
from fuse.dl.models.heads.heads_3D import Head3DRegression

from fuseimg.datasets.knight import KNIGHT
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault

from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC, MetricAccuracy, MetricConfusion
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
import torch.optim as optim
import fuse.utils.gpu as GPU
from fuse.utils.rand.seed import Seed
import logging
import time
import copy
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.lightning.pl_module import LightningModuleDefault
import pytorch_lightning as pl
# from nnunet.network_architecture.generic_UNet import Generic_UNet
# from nnunet.network_architecture.initialization import InitWeights_He
# from torchvision.models import resnet50

# from resnet import resnet34, resnet18, resnet50

from fuse.dl.models.heads.heads_1D import Head1D
## Parameters:
##############################################################################
# Data sources to use in model. Set {'imaging': True, 'clinical': False} for imaging only setting,
# and vice versa, or set both to True to use both.
# allocate gpus
# uncomment if you want to use specific gpus instead of automatically looking for free ones


def make_model(use_data: dict, num_classes: int, imaging_dropout: float, fused_dropout: float,
 pretrained:bool = False, regression_head: bool = False, two_dim: bool = False):
    if use_data["imaging"]:
        if two_dim:
            backbone = resnet50(True).float()
            backbone.fc = nn.Identity()
            conv_inputs = [("model.backbone_features", 2048)]
        elif not pretrained:
            backbone = BackboneResnet3D(in_channels=1)
            # state_dict = torch.load(open("/projects/msieve_dev3/usr/il018850/new_age_classification/base_resized/last.ckpt", "rb"))["state_dict"]
            # state_dict = {".".join(k.split(".")[2:]) : v for k, v in state_dict.items() if k.startswith("_model.backbone.")}
            # backbone.load_state_dict(state_dict)
            conv_inputs = [("model.backbone_features", 512)]
        else:
            # unet_path = "/data/usr/liam/nnUNet/3d_fullres/Task135_KiTS2021/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model"
            # norm_op_kwargs = {'eps': 1e-05, 'affine': True}
            # dropout_op_kwargs = {'p': 0, 'inplace': True}
            # net_nonlin_kwargs = {'negative_slope': 0.01, 'inplace': True}
            # backbone = Generic_UNet(1, 32, 4, 5, 2, 2, nn.Conv3d, nn.InstanceNorm3d, norm_op_kwargs, nn.Dropout3d,dropout_op_kwargs,
            #             nn.LeakyReLU, net_nonlin_kwargs, True, False, None, InitWeights_He(1e-2),
            #             [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            #             [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]] 
            #             , True, True, True)
            # loaded_state_dict = torch.load(unet_path)["state_dict"]
            # backbone.load_state_dict(state_dict=loaded_state_dict)
            # backbone.inference_apply_nonlin = None
            # conv_inputs = [('model.backbone_features', 320)]
            state_dict = torch.load(open("/data/usr/liam/age_classification/pretrain/resnet_50_23dataset.pth", "rb"))["state_dict"]
            state_dict = {k[7:]:v for k,v in state_dict.items()}
            backbone = resnet50()#(shortcut_type='A')
            backbone.load_state_dict(state_dict=state_dict)
            conv_inputs = [("model.backbone_features", 2048)]
    else:
        backbone = nn.Identity()
        conv_inputs = None
    if use_data["clinical"]:
        append_features = [("data.input.clinical.all", 11)]
    else:
        append_features = None

    if two_dim:
        heads = [
            Head1D(
                head_name="head_0",
                mode = "classification",
                conv_inputs=conv_inputs,
                dropout_rate=imaging_dropout,
                num_outputs=num_classes,
                append_features=append_features,
                append_layers_description=(256, 128),
            ),
        ]
    else:
        heads=[
                Head3DClassifier(
                    head_name="head_0",
                    # mode = "classification",
                    conv_inputs=conv_inputs,
                    dropout_rate=imaging_dropout,
                    # num_outputs=num_classes,
                    num_classes=num_classes,
                    append_features=append_features,
                    append_layers_description=(256, 128),
                    fused_dropout_rate=fused_dropout,
                ),
            ]
    if regression_head:
        if two_dim:
            heads.append(Head1D(
                head_name="head_1",
                mode="regression",
                conv_inputs=conv_inputs,
                dropout_rate=imaging_dropout,
                append_features=None,
                append_layers_description=(256, 128),
                num_outputs=1
                ))
        else:
            heads.append(Head3DRegression(
                head_name="head_1",
                # mode="regression",
                conv_inputs=conv_inputs,
                dropout_rate=imaging_dropout,
                append_features=None,
                append_layers_description=(256, 128),
                num_outputs=1
                ))
    model = ModelMultiHead(
        conv_inputs=(("data.input.img", 1),),
        backbone=backbone,
        heads=heads
    )
    return model


def main(cfg_path):
    # read config params
    cfg = yaml.safe_load(open(cfg_path))
    task_num = cfg['task_num']
    num_classes = cfg[task_num]["num_classes"]
    target_name = cfg[task_num]["target_name"]
    # read train/val splits file. for convenience, we use the one
    # auto-generated by the nnU-Net framework for the KiTS21 data
    dir_path = pathlib.Path(__file__).parent.resolve()
    splits = pd.read_pickle(os.path.join(dir_path, "splits_final.pkl"))
    # For this example, we use split 0 out of the 5 available cross validation splits
    split = splits[0]

    # read environment variables for data, cache and results locations
    data_path = os.environ["KNIGHT_DATA"]
    cache_path = os.path.join(os.environ["KNIGHT_CACHE"], str(cfg["experiment_num"]))
    results_path = os.environ["KNIGHT_RESULTS"]

    ## Basic settings:
    ##############################################################################
    # create model results dir:
    # we use a time stamp in model directory name, to prevent re-writing
    model_dir = os.path.join(results_path, cfg["exp_name"])
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # start logger
    fuse_logger_start(output_path=model_dir, console_verbose_level=logging.INFO)
    print("Done")

    # set constant seed for reproducibility.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for pytorch deterministic mode
    rand_gen = Seed.set_seed(1234, deterministic_mode=True)

    # select gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpus"][0])
    GPU.choose_and_enable_multiple_gpus(1, force_gpus=cfg["gpus"])

    ## Model definition
    ##############################################################################

    model = make_model(cfg["use_data"], num_classes, cfg["imaging_dropout"],
     cfg["fused_dropout"], pretrained=cfg["pretrained"],
      regression_head=cfg["regression_head"], two_dim=cfg["two_dim"])

    ## FuseMedML dataset preparation
    ##############################################################################

    train_ds, valid_ds = KNIGHT.dataset(
        data_path=data_path,
        cache_dir=cache_path,
        split=split,
        reset_cache=cfg["reset_cache"],
        resize_to=cfg["resize_to"],
        two_dim=cfg["two_dim"],
    )

    ## Create dataloader

    sampler = BatchSamplerDefault(
        dataset=train_ds,
        balanced_class_name=target_name,
        num_balanced_classes=num_classes,
        batch_size=cfg["batch_size"],
        balanced_class_weights=[1.0 / num_classes] * num_classes if task_num == "task_2" else None,
        mode="approx"
    )

    train_dl = DataLoader(
        dataset=train_ds,
        shuffle=False,
        drop_last=False,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=8,
    )

    valid_dl = DataLoader(
        dataset=valid_ds,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=cfg["batch_size"],
        num_workers=8,
        collate_fn=CollateDefault(),
        generator=rand_gen,
    )

    # Loss definition:
    ##############################################################################
    losses = {
        "cls_loss": LossDefault(pred="model.logits.head_0", target=target_name, callable=F.cross_entropy, weight=1.0),
    }
    if cfg["regression_head"]:
        losses["regression_loss"] = LossDefault(pred="model.output.head_1", target="data.aux_label", callable=nn.MSELoss(reduce=True))

    # Metrics definition:
    ##############################################################################
    train_metrics = OrderedDict(
        [
            ("op", MetricApplyThresholds(pred="model.output.head_0")),  # will apply argmax
            ("auc", MetricAUCROC(pred="model.output.head_0", target=target_name)),
            ("accuracy", MetricAccuracy(pred="results:metrics.op.cls_pred", target=target_name)),
            (
                "sensitivity",
                MetricConfusion(pred="results:metrics.op.cls_pred", target=target_name, metrics=("sensitivity",)),
            ),
        ]
    )
    val_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    best_epoch_source = dict(
        monitor=cfg[task_num]["target_metric"],  # can be any key from losses or metrics dictionaries
        mode="max",  # can be either min/max
    )

    # Optimizer definition:
    ##############################################################################
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.001)

    # Scheduler definition:
    ##############################################################################
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    ## Training
    ##############################################################################

    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=model_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )
    # create lightining trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=model_dir,
        max_epochs=cfg["num_epochs"],
        accelerator="gpu",
        devices=cfg["num_gpus"],
        strategy=None,
        auto_select_gpus=True,
        num_sanity_val_steps=-1,
    )

    # train
    pl_trainer.fit(pl_module, train_dl, valid_dl, ckpt_path=None)


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    main(config_path)
