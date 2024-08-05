from fuse.utils.utils_logger import fuse_logger_start
import os
from glob import glob
import pandas as pd
from fuse.dl.models import ModelMultiHead
from fuse.dl.models.backbones.backbone_resnet_3d import BackboneResnet3D
from fuse.dl.models.backbones.backbone_resnet import BackboneResnet
# from resnet import resnet18, resnet50
from fuse.dl.models.heads.heads_3D import Head3D

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
from fuse.data.utils.collates import CollateDefault

from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC, MetricAccuracy
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.metrics_common import MetricDefault
import torch.optim as optim
from fuse.utils.rand.seed import Seed
import logging
import sys
import copy
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.lightning.pl_module import LightningModuleDefault
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import hydra
from omegaconf import DictConfig
from clearml import Task
sys.path.append("/dccstor/mm_hcls2/oai/code")
from data.oai_ds import OAI
from fuse.dl.models.backbones.backbone_unet3d import UNet3D
from sklearn.metrics import mean_absolute_error

torch.set_float32_matmul_precision('medium')
torch.use_deterministic_algorithms(False)

@hydra.main(version_base="1.2", config_path=".", config_name="downstream_config")
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg)
    results_path = cfg.results_dir

    model_dir = os.path.join(results_path, cfg.experiment)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    # set constant seed for reproducibility.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for pytorch deterministic mode
    rand_gen = Seed.set_seed(1234, deterministic_mode=True)

    reg_targets = cfg.reg_targets
    cls_targets = cfg.cls_targets
    tags = [cfg.task, cfg.backbone]
    print(f"EXPERIMENT : {cfg.experiment}")
    if cfg.clearml:
        if len(glob(model_dir + "/*.cmlid")) == 0:
            task = Task.create(project_name=cfg.clearml_project_name, task_name=cfg.experiment)
            with open(os.path.join(model_dir, f"{task.id}.cmlid"), "w"):
                pass

        clearml_task_id = glob(model_dir + "/*.cmlid")[0].split("/")[-1].split(".")[0]
        task = Task.init(project_name=cfg.clearml_project_name, task_name=cfg.experiment, reuse_last_task_id=clearml_task_id,
                            continue_last_task=0, tags=tags)

        task.connect(cfg)


    elif cfg.backbone == "res3d":
        backbone = BackboneResnet3D(in_channels=1, pool=True, pretrained=cfg.pretrained)
        # backbone = Backbone_SwinTransformer()
        conv_inputs = [("model.backbone_features", 512)]

    elif cfg.backbone == "s3d":
        backbone2d = BackboneResnet(in_channels = 1, name = "resnet18", pretrained=cfg.pretrained)
        if cfg.load_ckpt:
            state_dict = torch.load(open(cfg.ckpt_path, "rb"), map_location=torch.device('cpu'))
            state_dict = {k.replace("module.backbone.", "") :v for k, v in state_dict["teacher"].items() if "module.backbone." in k}

            # state_dict = {k.replace("module.torch_model.backbone_1.", "") :v for k, v in state_dict["student"].items() if "backbone_1." in k}
            # state_dict = {k.replace("_model.1.model.backbone_1.", "") :v for k, v in state_dict["state_dict"].items() if "backbone_1." in k}

            backbone2d.load_state_dict(state_dict)

        # for name, para in backbone2d.named_parameters():
        #     para.requires_grad = False
        
        context3d = Model3DContext(512)
        backbone = ModelSliced3D(backbone2d, context3d, pool=True)
        conv_inputs = [("model.backbone_features", 512)]

    elif cfg.backbone == "swinunet":
        backbone = SwinUNETR(96, 1, 3, feature_size=48, only_encoder=True)

        if os.path.exists(cfg.ckpt_path):
            print("not implemented yet!!!!")
        elif cfg.pretrained:
            state_dict = torch.load("/dccstor/mm_hcls2/oai/code/archs/weights/self_supervised_nv_swin_unetr_50000.pth",map_location=torch.device('cpu'))
            state_dict = {k.replace("encoder.","swinViT.").replace("0.0.","0.") :v for k,v in state_dict["model"].items()}
        else:
            state_dict = backbone.state_dict()
        
        backbone.load_state_dict(state_dict, strict=False)
        conv_inputs = [("model.backbone_features", 768)]

    elif cfg.backbone == "unet3d":
        backbone = UNet3D(for_cls=True)

        if os.path.exists(cfg.ckpt_path):
            print(F"LOADING ckpt from {cfg.ckpt_path}")
            state_dict = torch.load(open(cfg.ckpt_path, "rb"), map_location=torch.device('cpu'))
            state_dict = {k.replace("teacher_backbone.",""):v for k,v in state_dict["state_dict"].items() if "teacher_backbone." in k}
        elif cfg.pretrained:
            state_dict = torch.load("/dccstor/mm_hcls2/oai/code/archs/weights/supervised_suprem_unet_2100.pth",map_location=torch.device('cpu'))
            state_dict = {k.replace("module.backbone.",""):v for k,v in state_dict["net"].items()}
        else:
            state_dict = backbone.state_dict()
        
        backbone.load_state_dict(state_dict, strict=False)
        conv_inputs = [("model.backbone_features", 512)]

    ## FuseMedML dataset preparation
    ##############################################################################
    columns_to_extract = ["accession_number", "path"] + cfg.reg_targets + cfg.cls_targets
    train_dls = []
    val_dls = []
    if len(reg_targets) > 0:
        reg_df = pd.read_csv(cfg.csv_path)
        reg_df = reg_df[~reg_df[cfg.reg_targets[0]].isna()]
        reg_train_df = reg_df[reg_df.fold.isin(list(range(16)))]
        reg_val_df = reg_df[reg_df.fold.isin(list(range(16,20)))]
        reg_targets = {name : 
                    {"mean": reg_df[name].mean(), "std": reg_df[name].std()} for name in reg_targets
                    }
        print(reg_targets)
        for reg_target, stats in reg_targets.items():
            reg_train_df[reg_target] = reg_train_df[reg_target].apply(lambda x: (x-stats["mean"])/stats["std"])
            reg_val_df[reg_target] = reg_val_df[reg_target].apply(lambda x: (x-stats["mean"])/stats["std"])
        print(f"reg train samples = {len(reg_train_df)}")
        print(f"reg val samples = {len(reg_val_df)}")

        reg_train_ds = OAI.dataset(reg_train_df, for_classification=True, validation=True, columns_to_extract=columns_to_extract, resize_to=cfg.resize_to)
        reg_val_ds = OAI.dataset(reg_val_df, for_classification=True, validation=True, columns_to_extract=columns_to_extract, resize_to=cfg.resize_to)

        reg_train_dl = DataLoader(
            dataset=reg_train_ds,
            shuffle=True,
            drop_last=False,
            # batch_sampler=sampler,
            batch_size=cfg.batch_size,
            collate_fn=CollateDefault(),
            num_workers=cfg.n_workers,
        )
        train_dls.append(reg_train_dl)
        reg_val_dl = DataLoader(
            dataset=reg_val_ds,
            shuffle=False,
            drop_last=False,
            batch_sampler=None,
            batch_size=cfg.batch_size,
            num_workers=cfg.n_workers,
            collate_fn=CollateDefault(),
            generator=rand_gen,
        )
        val_dls.append(reg_val_dl)

    heads=[
            Head3D(
                head_name=f"head_{target}",
                mode="regression",
                conv_inputs=conv_inputs,
                num_outputs=1,
            ) for target in reg_targets
            ]


    if len(cls_targets) > 0:
        cls_df = pd.read_csv(cfg.csv_path)
        if "V00COHORT" in cls_targets:
            cls_df = cls_df[cls_df.V00COHORT != "2: Incidence"]
        cls_train_df = cls_df[cls_df.fold.isin(list(range(16)))]
        cls_val_df = cls_df[cls_df.fold.isin(list(range(16,20)))]
        cls_targets = {name : 
                    {"classes" : list(cls_train_df[name].unique()), **dict(cls_train_df[name].value_counts(normalize=True))}
                        for name in cls_targets
                    }
        print(cls_targets)
        for cls_target, stats in cls_targets.items():
            cls_train_df[cls_target] = cls_train_df[cls_target].apply(lambda x: stats["classes"].index(x))
            cls_val_df[cls_target] = cls_val_df[cls_target].apply(lambda x: stats["classes"].index(x))

        print(f"cls train samples = {len(cls_train_df)}")
        print(f"cls val samples = {len(cls_val_df)}")
        cls_train_ds = OAI.dataset(cls_train_df, for_classification=True, validation=True, columns_to_extract=columns_to_extract, resize_to=cfg.resize_to)
        cls_val_ds = OAI.dataset(cls_val_df, for_classification=True, validation=True, columns_to_extract=columns_to_extract, resize_to=cfg.resize_to)

        cls_train_dl = DataLoader(
            dataset=cls_train_ds,
            shuffle=True,
            drop_last=False,
            # batch_sampler=sampler,
            batch_size=cfg.batch_size,
            collate_fn=CollateDefault(),
            num_workers=cfg.n_workers,
        )
        train_dls.append(cls_train_dl)
        cls_val_dl = DataLoader(
            dataset=cls_val_ds,
            shuffle=False,
            drop_last=False,
            batch_sampler=None,
            batch_size=cfg.batch_size,
            num_workers=cfg.n_workers,
            collate_fn=CollateDefault(),
            generator=rand_gen,
        )
        val_dls.append(cls_val_dl)
        heads.extend([
            Head3D(
                    head_name=f"head_{target}",
                    mode="classification",
                    conv_inputs=conv_inputs,
                    num_outputs=len(stats["classes"]),
                ) for target, stats in cls_targets.items()
        ])
    
    model = ModelMultiHead(
        conv_inputs=(("img", 1),),
        backbone=backbone,
        heads=heads
    )

    # read environment variables for data, cache and results locations



    ## Basic settings:
    ##############################################################################
    # create model results dir:
    # we use a time stamp in model directory name, to prevent re-writing


    # start logger
    fuse_logger_start(output_path=model_dir, console_verbose_level=logging.INFO)
    print("Done")




    


    # Loss definition:
    ##############################################################################
    losses = {
        f"loss_{target}": LossDefault(pred=f"model.output.head_{target}", target=target, callable=nn.MSELoss(), weight=1.0) for target in reg_targets
    }
    if len(cls_targets) > 0:
        weights = {cls_target : torch.tensor([1/stats[cls] for cls in stats["classes"]], device=torch.device('cuda'), dtype=torch.float32) for cls_target, stats in cls_targets.items()}
        for cls_target, stats in cls_targets.items():
            losses[f"loss_{cls_target}"] = LossDefault(pred=f"model.output.head_{cls_target}", target=cls_target, callable=nn.CrossEntropyLoss(weight=weights[cls_target]), weight=1.0)

    # Metrics definition:
    ##############################################################################
    train_metrics = {}

    def my_mae_func(pred,target):
        return mean_absolute_error(target,pred)


    val_metrics = {
            f"mae_{target}": MetricDefault(metric_func=my_mae_func,pred=f"model.output.head_{target}", target=target) for target in reg_targets
        }
    if len(cls_targets) > 0:
        for cls_target, stats in cls_targets.items():
            val_metrics[f"op_{cls_target}"] = MetricApplyThresholds(pred=f"model.output.head_{cls_target}")# will apply argmax
            val_metrics[f"auc_{cls_target}"] = MetricAUCROC(pred=f"model.output.head_{cls_target}", target=cls_target, class_names=stats["classes"] if len(stats["classes"])>2 else None)
            val_metrics[f"acc_{cls_target}"] = MetricAccuracy(pred=f"results:metrics.op_{cls_target}.cls_pred",target=cls_target,)
            train_metrics[f"op_{cls_target}"] = MetricApplyThresholds(pred=f"model.output.head_{cls_target}")# will apply argmax
            train_metrics[f"auc_{cls_target}"] = MetricAUCROC(pred=f"model.output.head_{cls_target}", target=cls_target, class_names=stats["classes"] if len(stats["classes"])>2 else None)
            train_metrics[f"acc_{cls_target}"] = MetricAccuracy(pred=f"results:metrics.op_{cls_target}.cls_pred",target=cls_target,)

    # best_epoch_source = dict(
    #     monitor="validation.metrics.auc.macro_avg",  # can be any key from losses or metrics dictionaries
    #     mode="max",  # can be either min/max
    # )

    # Optimizer definition:
    ##############################################################################

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


    # Scheduler definition:
    ##############################################################################
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    lr_sch_config = dict(scheduler=lr_scheduler)
    optimizers_and_lr_schs = dict(optimizer=optimizer)
    optimizers_and_lr_schs["lr_scheduler"] = lr_sch_config

    #CallBacks

    callbacks = [
        # BackboneFinetuning(unfreeze_backbone_at_epoch=8,),
        LearningRateMonitor(logging_interval='epoch'),
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
        test_metrics=copy.deepcopy(val_metrics),
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
        gradient_clip_val= cfg.grad_clip,
        deterministic=False,
        precision=cfg.precision,
        # enable_checkpointing=False,
    )

    # train
    pl_trainer.fit(pl_module, train_dls[0], val_dls[0], ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
