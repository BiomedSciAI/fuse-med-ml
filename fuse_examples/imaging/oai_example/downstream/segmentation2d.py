import logging
import os
from glob import glob
from typing import Any

import dill
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
from clearml import Task
from monai.losses import DiceLoss
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data.dataloader import DataLoader

from fuse.data.utils.collates import CollateDefault
from fuse.dl.lightning.pl_funcs import (
    step_extract_predictions,
    step_losses,
    step_metrics,
)
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models import ModelMultiHead
from fuse.dl.models.backbones.backbone_effiunet import Effi_UNet
from fuse.eval.metrics.segmentation.metrics_segmentation_common import MetricDice
from fuse.utils.rand.seed import Seed
from fuse.utils.utils_logger import fuse_logger_start
from fuse_examples.imaging.oai_example.data.seg_ds import SegOAI

torch.set_float32_matmul_precision("medium")
torch.use_deterministic_algorithms(False)


def remove_blobs(pred_tensor: torch.Tensor) -> torch.Tensor:
    # FILL with post-processing algorithm on the predicted segmentation
    return pred_tensor


@hydra.main(version_base="1.2", config_path=".", config_name="segmentation_config")
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg)
    results_path = cfg.results_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cfg.cuda_devices])

    assert (
        sum(
            cfg[weights] is not None
            for weights in [
                "baseline_weights",
                "dino_weights",
                "mae_weights",
                "resume_training_from",
                "test_ckpt",
            ]
        )
        <= 1
    ), "only one weights/ckpt path can be used at a time"

    model_dir = os.path.join(results_path, cfg.experiment)
    cache_dir = os.path.join(results_path, cfg.task)

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

        backbone = Effi_UNet(
            cfg.backbone,
            in_channels=1,
            classes=cfg.num_classes,
            encoder_weights="imagenet",
        )
        if cfg.dino_weights is not None:
            print(f"LOADING ckpt from {cfg.dino_weights}")
            state_dict = torch.load(
                open(cfg.dino_weights, "rb"), map_location=torch.device("cpu")
            )
            state_dict = {
                k.replace("student_backbone", "encoder"): v
                for k, v in state_dict["state_dict"].items()
                if "student_backbone." in k
            }
        elif cfg.baseline_weights is not None:
            pass
        elif cfg.mae_weights is not None:
            state_dict = torch.load(cfg.mae_weights, map_location=torch.device("cpu"))
            state_dict = {
                k.replace("_model.backbone.", ""): v
                for k, v in state_dict["state_dict"].items()
                if "classifier" not in k
            }

        else:
            backbone = Effi_UNet(
                cfg.backbone,
                in_channels=1,
                classes=cfg.num_classes,
                encoder_weights=None,
            )
            state_dict = backbone.state_dict()

        backbone.load_state_dict(state_dict, strict=False)

    heads = []

    model = ModelMultiHead(conv_inputs=(("img", 1),), backbone=backbone, heads=heads)

    # read environment variables for data, cache and results locations

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
    df_all = pd.read_csv(cfg.csv_path)
    # create row for each of the slices
    dfs = {}
    dfs["train"] = df_all[df_all.fold.isin(cfg.train_folds)]
    dfs["train"] = pd.concat([dfs["train"].assign(slice=i) for i in range(160)])
    dfs["train"]["idx"] = range(len(dfs["train"]))

    dfs["val"] = df_all[df_all.fold.isin(cfg.val_folds)]
    dfs["val"]["idx"] = range(len(dfs["val"]))

    dfs["test"] = df_all[df_all.fold.isin(cfg.test_folds)]
    dfs["test"]["idx"] = range(len(dfs["test"]))

    print(f"train samples = {len(dfs['train'])}")
    print(f"val samples = {len(dfs['val'])}")

    train_ds = SegOAI.dataset(
        dfs["train"],
        cache_dir=f"{cache_dir}/train_cache_samples",
        validation=(not cfg.aug),
        resize_to=None,
        num_classes=cfg.num_classes,
        im2D=True,
        num_workers=cfg.n_workers,
    )
    val_ds = SegOAI.dataset(
        dfs["val"],
        validation=True,
        resize_to=None,
        num_classes=cfg.num_classes,
        im2D=False,
        num_workers=cfg.n_workers // 8,
    )
    test_ds = SegOAI.dataset(
        dfs["test"],
        validation=True,
        resize_to=None,
        num_classes=cfg.num_classes,
        im2D=False,
        num_workers=cfg.n_workers // 8,
    )
    # test_ds = SegOAI.dataset(dfs["test"], validation=True, resize_to=None, num_classes=cfg.num_classes, im2D=True)
    ## Create dataloader

    train_dl = DataLoader(
        dataset=train_ds,
        shuffle=True,
        drop_last=False,
        # batch_sampler=sampler,
        batch_size=cfg.batch_size,
        collate_fn=CollateDefault(),
        num_workers=cfg.n_workers,
    )

    valid_dl = DataLoader(
        dataset=val_ds,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=1,
        num_workers=cfg.n_workers // 8,
        collate_fn=CollateDefault(),
        generator=rand_gen,
    )
    test_dl = DataLoader(
        dataset=test_ds,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=1,
        num_workers=cfg.n_workers // 8,
        collate_fn=CollateDefault(),
        generator=rand_gen,
    )

    # Loss definition:
    ##############################################################################
    # callable_loss = GeneralizedDiceLoss(include_background=cfg.include_background,  sigmoid=cfg.sigmoid, softmax=cfg.softmax)
    callable_loss = DiceLoss(
        include_background=cfg.include_background,
        sigmoid=cfg.sigmoid,
        softmax=cfg.softmax,
    )
    losses = {
        "generalized_dice_loss": LossDefault(
            pred="model.backbone_features",
            target="seg",
            callable=callable_loss,
            weight=1.0,
        )
    }

    # Metrics definition:
    ##############################################################################
    def pre_process_for_dice(sample: dict) -> dict:
        sample["model.backbone_features"] = sample["model.backbone_features"].argmax(
            axis=0
        )
        sample["model.backbone_features"] = remove_blobs(
            sample["model.backbone_features"]
        )
        sample["seg"] = sample["seg"].argmax(axis=0)
        return sample

    train_metrics = {}
    val_metrics = {
        "dice": MetricDice(
            pred="model.backbone_features",
            target="seg",
            pre_collect_process_func=pre_process_for_dice,
        )
    }

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
    # lr_scheduler = cosine_annealing_with_warmup_lr_scheduler(optimizer, num_warmup_steps=cfg.num_warmup_steps, T_max=cfg.n_epochs, eta_min=0.00001)
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

    def validation_step(
        self: Any, batch_dict: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        # add step number to batch_dict
        batch_dict["global_step"] = self.global_step
        batch_dict["img"] = (
            batch_dict["img"].squeeze(0).permute(1, 0, 2, 3)
        )  # (160,1,384,384)
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict = self.forward(batch_dict)  # [1, 7, 160, 384, 384]
        batch_dict["model.backbone_features"] = (
            batch_dict["model.backbone_features"].permute(1, 0, 2, 3).unsqueeze(0)
        )
        batch_dict["model.backbone_features"] = remove_blobs(
            batch_dict["model.backbone_features"]
        )
        if self._validation_losses is not None:
            losses = self._validation_losses[dataloader_idx][1]
        else:
            losses = self._losses
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss (ignored) and save losses values in batch_dict["losses"]
        _ = step_losses(losses, batch_dict)
        # given the batch_dict and FuseMedML style metrics - collect the required values to compute the metrics on epoch_end
        step_metrics(self._validation_metrics[dataloader_idx][1], batch_dict)
        # aggregate losses
        if losses:  # if there are losses, collect the results
            self._validation_step_outputs[dataloader_idx].append(
                {"losses": batch_dict["losses"]}
            )

    pl_module.validation_step = validation_step.__get__(pl_module)

    # create lightining trainer.
    devices = torch.cuda.device_count()
    pl_trainer = pl.Trainer(
        default_root_dir=model_dir,
        max_epochs=cfg.n_epochs,
        accelerator="gpu",
        devices=devices,
        strategy="ddp" if devices > 1 else "auto",
        num_sanity_val_steps=0,
        gradient_clip_val=cfg.grad_clip,
        deterministic=False,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        precision=cfg.precision,
        # enable_checkpointing=False,
        # limit_train_batches=3,
    )
    if cfg.test_ckpt is not None:

        print(f"Test using ckpt: {cfg.test_ckpt}")
        pl_trainer.validate(pl_module, test_dl, ckpt_path=cfg.test_ckpt)
        if cfg.save_test_results:

            def predict_step(self: Any, batch_dict: dict, batch_idx: int) -> dict:
                batch_dict["img"] = (
                    batch_dict["img"].squeeze(0).permute(1, 0, 2, 3)
                )  # [160,1,384,384]
                batch_dict = self.forward(batch_dict)
                batch_dict["model.backbone_features"] = (
                    batch_dict["model.backbone_features"]
                    .permute(1, 0, 2, 3)
                    .argmax(axis=0)
                    .to(torch.uint8)
                )  # [1,160,384,384]
                return step_extract_predictions(self._prediction_keys, batch_dict)

            pl_module.predict_step = predict_step.__get__(pl_module)
            pl_module.set_predictions_keys(["model.backbone_features", "idx"])
            output = pl_trainer.predict(pl_module, test_dl, ckpt_path=cfg.test_ckpt)
            with open(f"{model_dir}/output.pkl", "wb") as f:
                dill.dump(output, f, protocol=4)
    else:
        print(f"Training using ckpt: {ckpt_path}")
        pl_trainer.fit(pl_module, train_dl, valid_dl, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
