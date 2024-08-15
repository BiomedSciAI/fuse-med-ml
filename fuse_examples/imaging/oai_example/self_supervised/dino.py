# This file is a modifies version of the Dino example from lightly - https://github.com/lightly-ai/lightly/blob/master/examples/pytorch_lightning/dino.py

import copy
import torch
import pytorch_lightning as pl
import pandas as pd
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from typing import Any
import os
from glob import glob
from fuse.data.utils.collates import CollateDefault
from fuse.dl.models.backbones.backbone_vit import vit_base
from fuse.utils import NDict
from fuse.dl.models.backbones.backbone_resnet_3d import BackboneResnet3D
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from fuse.dl.models.backbones.backbone_unet3d import UNet3D
from fuse_examples.imaging.oai_example.data.oai_ds import OAI
from omegaconf import DictConfig
import hydra
from clearml import Task

torch.set_float32_matmul_precision("high")


class DINO(pl.LightningModule):
    def __init__(self, cfg: Any):
        super().__init__()

        # self.automatic_optimization=False
        # self.save_hyperparameters(cfg)
        if cfg.backbone == "res18":
            backbone = BackboneResnet3D(pretrained=False, in_channels=1, pool=True)
            input_dim = 512
        elif cfg.backbone == "vit":
            backbone = vit_base(cfg.resize_to, cfg.patch_shape, 1)
            input_dim = 768
        elif cfg.backbone == "unet3d":
            backbone = UNet3D(for_cls=True)
            if cfg.suprem_weights is not None:
                state_dict = torch.load(
                    cfg.suprem_weights, map_location=torch.device("cpu")
                )
                state_dict = {
                    k.replace("module.backbone.", ""): v
                    for k, v in state_dict["net"].items()
                }
                backbone.load_state_dict(state_dict, strict=False)

            def hook(
                module: torch.nn.Module, input: Any, output: torch.Tensor
            ) -> torch.Tensor:
                return torch.nn.AdaptiveMaxPool3d(output_size=1)(output).squeeze()

            backbone.register_forward_hook(hook)
            input_dim = 512

        self.student_backbone = backbone

        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.current_step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        y = self.teacher_backbone(x).flatten(start_dim=1)

        z = self.teacher_head(y)
        return z, y

    def training_step(self, batch_dict: NDict, batch_idx: int) -> torch.Tensor:
        momentum = cosine_schedule(
            self.current_epoch, self.cfg.n_epochs, self.cfg.momentum_teacher, 1
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        views = [batch_dict[f"crop_{i}"] for i in range(self.cfg.n_crops)]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]

        teacher_out, t_embs = zip(
            *[self.forward_teacher(view) for view in global_views]
        )
        student_out = [self.forward(view) for view in views]
        dino_loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        if self.cfg.clearml:
            self.cfg.logger.report_scalar(
                "dino_loss",
                "dino_loss",
                iteration=self.current_step,
                value=dino_loss.item(),
            )
        self.log("dino_loss", dino_loss.item())

        return dino_loss

    def on_after_backward(self) -> None:
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def validation_step(self, batch_dict: NDict, batch_idx: int) -> torch.Tensor:
        views = [batch_dict[f"crop_{i}"] for i in range(self.cfg.n_crops)]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]

        teacher_out, t_embs = zip(
            *[self.forward_teacher(view) for view in global_views]
        )
        # if (self.current_epoch+1) % 8 != 0:
        student_out = [self.forward(view) for view in views]
        dino_loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        if self.cfg.clearml:
            self.cfg.logger.report_scalar(
                "dino_loss",
                "val_dino_loss",
                iteration=self.current_step,
                value=dino_loss.item(),
            )
        self.log("val_dino_loss", dino_loss.item())

        return dino_loss

    def configure_optimizers(self) -> tuple:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
        )
        # lr_scheduler = {"scheduler" :torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=9),
        # "monitor" : "dino_loss", "interval" : "epoch"}
        lr_scheduler = {
            "scheduler": CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.n_epochs)
        }
        return [optimizer], [lr_scheduler]


@hydra.main(version_base="1.2", config_path=".", config_name="dino_config")
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cfg.cuda_devices])

    assert (
        sum(
            cfg[weights] is not None
            for weights in ["suprem_weights", "resume_training_from"]
        )
        <= 1
    ), "only one weights/ckpt path can be used at a time"

    device_count = torch.cuda.device_count()
    logging_path = os.path.join(cfg.model_dir, cfg.experiment)
    cfg.logging_path = logging_path
    if not os.path.isdir(logging_path):
        os.makedirs(logging_path)

    if cfg.clearml:
        if len(glob(logging_path + "/*.cmlid")) == 0:
            task = Task.create(
                project_name=cfg.clearml_project_name, task_name=cfg.experiment
            )
            with open(os.path.join(logging_path, f"{task.id}.cmlid"), "w"):
                pass

        clearml_task_id = (
            glob(logging_path + "/*.cmlid")[0].split("/")[-1].split(".")[0]
        )
        task = Task.init(
            project_name="OAI/Dino",
            task_name=cfg.experiment,
            reuse_last_task_id=clearml_task_id,
            continue_last_task=0,
        )

        task.connect(cfg)

        logger = task.get_logger()
        cfg.logger = logger

    all_data_df = pd.read_csv(cfg.csv_path)
    train_split = all_data_df[all_data_df["fold"].isin(cfg.train_folds)]
    val_split = all_data_df[all_data_df["fold"].isin(cfg.val_folds)]

    train_ds = OAI.dataset(
        train_split,
        for_classification=False,
        resize_to=cfg.resize_to,
        n_crops=cfg.n_crops,
        num_workers=cfg.n_workers,
    )
    val_ds = OAI.dataset(
        val_split,
        for_classification=False,
        resize_to=cfg.resize_to,
        n_crops=cfg.n_crops,
        num_workers=cfg.n_workers,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.n_workers,
        collate_fn=CollateDefault(),
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.n_workers,
        collate_fn=CollateDefault(),
        pin_memory=True,
    )
    cfg.len_dl = len(train_dl)

    if cfg.resume_training_from is not None:
        ckpt_path = cfg.resume_training_from
    elif len(glob(f"{logging_path}/*.ckpt")) > 0:
        ckpt_path = sorted(glob(f"{logging_path}/*.ckpt"))[-1]
    else:
        ckpt_path = None

    print(f"\n LOADING WEIGHTS FROM ---{ckpt_path}--- \n")

    checkpoint_callback = ModelCheckpoint(
        dirpath=logging_path,
        filename="{epoch}-{step}",  # Customize the filename format (optional)
        save_top_k=2,  # Save the top 1 model with lowest loss
        monitor="val_dino_loss",  # Monitor validation loss
        mode="min",  # Save the model with the lowest validation loss
        save_last=True,  # Save the last checkpoint
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")  # Or adjust interval

    pl_module = DINO(cfg)
    pl_trainer = pl.Trainer(
        default_root_dir=logging_path,
        max_epochs=cfg.n_epochs,
        accelerator="gpu",
        devices=device_count,
        callbacks=[checkpoint_callback, lr_monitor],
        # num_nodes=int(os.environ.get('WORLD_SIZE',device_count))/device_count,
        # strategy="ddp",
        limit_train_batches=0.5,
        gradient_clip_val=cfg.clip_grad,
        num_sanity_val_steps=0,
        precision=cfg.precision,
    )

    # train
    pl_trainer.fit(pl_module, train_dl, val_dl, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
