import logging
import os
from glob import glob

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from clearml import Task
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data.dataloader import DataLoader

from fuse.data.utils.collates import CollateDefault
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models import ModelMultiHead
from fuse.dl.models.backbones.backbone_effiunet import Effi_UNet
from fuse.utils.rand.seed import Seed
from fuse.utils.utils_logger import fuse_logger_start
from fuse_examples.imaging.oai_example.data.oai_ds import OAI

torch.set_float32_matmul_precision("medium")
torch.use_deterministic_algorithms(False)
# process = subprocess.Popen("nvidia-smi".split(), stdout=subprocess.PIPE)
# output, error = process.communicate()
# print(output.decode("utf-8"))


@hydra.main(version_base="1.2", config_path=".", config_name="mae_config")
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cfg.cuda_devices])
    results_path = cfg.results_dir

    model_dir = os.path.join(results_path, cfg.experiment)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    tags = cfg.tags + [cfg.backbone]
    print(f"EXPERIMENT : {cfg.experiment}")
    if cfg.clearml:
        if len(glob(model_dir + "/*.cmlid")) == 0:
            task = Task.create(
                project_name=cfg.clearml_project_name, task_name=cfg.experiment
            )
            with open(os.path.join(model_dir, f"{task.id}.cmlid"), "w"):
                pass

        clearml_task_id = glob(model_dir + "/*.cmlid")[0].split("/")[-1].split(".")[0]
        task = Task.init(
            project_name=cfg.clearml_project_name,
            task_name=cfg.experiment,
            reuse_last_task_id=clearml_task_id,
            continue_last_task=0,
            tags=tags,
        )

        task.connect(cfg)
        task.add_tags(cfg.tags)

    if cfg.backbone.startswith("efficientnet"):
        weights = "imagenet" if cfg.pretrained else None
        backbone = Effi_UNet(
            cfg.backbone,
            in_channels=1,
            classes=1,
            encoder_weights=weights,
            activation="sigmoid",
        )

    heads = []

    model = ModelMultiHead(
        conv_inputs=(("masked_img", 1),), backbone=backbone, heads=heads
    )
    ## Basic settings:
    ##############################################################################
    # create model results dir:
    # we use a time stamp in model directory name, to prevent re-writing

    # start logger
    fuse_logger_start(output_path=model_dir, console_verbose_level=logging.INFO)
    print("Done")

    # set constant seed for reproducibility.
    os.environ[
        "CUBLAS_WORKSPACE_CONFIG"
    ] = ":4096:8"  # required for pytorch deterministic mode
    rand_gen = Seed.set_seed(1234, deterministic_mode=True)

    ## FuseMedML dataset preparation
    ##############################################################################
    all_data_df = pd.read_csv(cfg.csv_path)
    all_data_df = all_data_df[["accession_number", "path", "fold", "max_val"]]

    train_split = all_data_df[all_data_df["fold"].isin(cfg.train_folds)]
    val_split = all_data_df[all_data_df["fold"].isin(cfg.val_folds)]

    ds_train = OAI.dataset(
        train_split,
        # columns_to_extract=["accession_number", "path", "max_val"],
        for_classification=False,
        validation=False,
        mae_cfg=cfg.mae_cfg,
        resize_to=cfg.resize_to,
        num_workers=cfg.n_workers,
        im2D=True,
    )

    ds_val = OAI.dataset(
        val_split,
        # columns_to_extract=["accession_number", "path", "max_val"],
        for_classification=False,
        validation=True,
        mae_cfg=cfg.mae_cfg,
        resize_to=cfg.resize_to,
        num_workers=cfg.n_workers,
        im2D=True,
    )
    print(f"num of training samples: {len(ds_train)}")
    print(f"num of validation samples: {len(ds_val)}")
    ## Create dataloader

    train_dl = DataLoader(
        dataset=ds_train,
        shuffle=True,
        drop_last=False,
        # batch_sampler=sampler,
        batch_size=cfg.batch_size,
        collate_fn=CollateDefault(),
        num_workers=cfg.n_workers,
    )

    valid_dl = DataLoader(
        dataset=ds_val,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        collate_fn=CollateDefault(),
        generator=rand_gen,
    )

    # Loss definition:
    ##############################################################################
    losses = {
        "loss_mse": LossDefault(
            pred="model.backbone_features",
            target="img",
            callable=nn.MSELoss(reduction="sum"),
            weight=1.0,
        )
    }

    # Metrics definition:
    ##############################################################################
    train_metrics = {}
    val_metrics = {}

    # best_epoch_source = dict(
    #     monitor="validation.metrics.auc.macro_avg",  # can be any key from losses or metrics dictionaries
    #     mode="max",  # can be either min/max
    # )

    # Optimizer definition:
    ##############################################################################

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # Scheduler definition:
    ##############################################################################
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    lr_sch_config = dict(scheduler=lr_scheduler)
    optimizers_and_lr_schs = dict(optimizer=optimizer)
    optimizers_and_lr_schs["lr_scheduler"] = lr_sch_config

    # CallBacks

    callbacks = [
        # BackboneFinetuning(unfreeze_backbone_at_epoch=8,),
        LearningRateMonitor(logging_interval="epoch"),
        # ModelCheckpoint(dirpath=model_dir, save_last=True)
    ]

    ## Training
    ##############################################################################
    ckpt_path = None
    if os.path.isfile(os.path.join(model_dir, "last.ckpt")):
        ckpt_path = os.path.join(model_dir, "last.ckpt")
        print(f"LOADING CHECKPOINT FROM {ckpt_path}")

    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=model_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        test_metrics=val_metrics,
        # best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
        callbacks=callbacks,
        save_model=False,
        log_unit="epoch",
    )
    # create lightining trainer.
    devices = torch.cuda.device_count()
    pl_trainer = pl.Trainer(
        default_root_dir=model_dir,
        max_epochs=cfg.n_epochs,
        accelerator="gpu",
        devices=devices,
        strategy="ddp" if devices > 1 else "auto",
        num_sanity_val_steps=0,
        limit_train_batches=0.2,
        # limit_val_batches=0.2,
        gradient_clip_val=cfg.grad_clip,
        deterministic=False,
        # enable_checkpointing=False,
    )
    # train
    pl_trainer.fit(pl_module, train_dl, valid_dl, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
