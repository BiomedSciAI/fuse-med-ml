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

import copy
import logging
import os
from collections import OrderedDict
from collections.abc import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data.dataloader import DataLoader
from torchvision.models.resnet import ResNet18_Weights

import fuse.utils.gpu as GPU
from fuse.data.utils.collates import CollateDefault
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models import ModelMultiHead
from fuse.dl.models.backbones.backbone_inception_resnet_v2 import (
    BackboneInceptionResnetV2,
)
from fuse.dl.models.backbones.backbone_resnet import BackboneResnet
from fuse.dl.models.backbones.backbone_vit import ViT
from fuse.dl.models.heads.head_global_pooling_classifier import (
    HeadGlobalPoolingClassifier,
)
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.eval.evaluator import EvaluatorDefault
from fuse.eval.metrics.classification.metrics_classification_common import (
    MetricAccuracy,
    MetricAUCROC,
    MetricROCCurve,
)
from fuse.eval.metrics.classification.metrics_thresholding_common import (
    MetricApplyThresholds,
)
from fuse.utils.file_io.file_io import create_dir, load_pickle, save_dataframe
from fuse.utils.utils_debug import FuseDebug
from fuse.utils.utils_logger import fuse_logger_start
from fuse_examples.imaging.classification.isic.golden_members import FULL_GOLDEN_MEMBERS
from fuseimg.datasets.isic import ISIC, ISICDataModule

###########################################################################################################
# Fuse
###########################################################################################################
##########################################
# Debug modes
##########################################
mode = "default"  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################
# GPUs and Workers
##########################################
NUM_GPUS = 1  # supports multiple gpu training with DDP strategy
NUM_WORKERS = 8

##########################################
# Modality
##########################################
multimodality = (
    True  # Set: 'False' to use only imaging, 'True' to use imaging & meta-data
)

##########################################
# Model Type
##########################################
model_type = "Transformer"  # Set: 'Transformer' to use ViT/MMViT, 'CNN' to use InceptionResNet or Resnet18

##########################################
# Output Paths
##########################################


ROOT = "./_examples/isic/"
DATA = (
    os.environ["ISIC19_DATA_PATH"]
    if "ISIC19_DATA_PATH" in os.environ
    else os.path.join(ROOT, "data_dir")
)
modality = "multimodality" if multimodality else "imaging"
model_dir = os.path.join(ROOT, f"model_dir_{modality}")
PATHS = {
    "model_dir": model_dir,
    "inference_dir": os.path.join(model_dir, "infer_dir"),
    "eval_dir": os.path.join(model_dir, "eval_dir"),
    "data_dir": DATA,
    "cache_dir": os.path.join(ROOT, "cache_dir"),
    "data_split_filename": os.path.join(model_dir, "isic_split.pkl"),
}


##########################################
# Train Common Params
##########################################
TRAIN_COMMON_PARAMS = {}
# ============
# Data
# ============
TRAIN_COMMON_PARAMS[
    "data.batch_size"
] = 32  # effective batch size = batch_size * num_gpus
TRAIN_COMMON_PARAMS["data.num_workers"] = NUM_WORKERS
TRAIN_COMMON_PARAMS["data.num_folds"] = 5
TRAIN_COMMON_PARAMS["data.train_folds"] = [0, 1, 2]
TRAIN_COMMON_PARAMS["data.validation_folds"] = [3]
TRAIN_COMMON_PARAMS["data.infer_folds"] = [4]
TRAIN_COMMON_PARAMS["data.samples_ids"] = {"all": None, "golden": FULL_GOLDEN_MEMBERS}[
    "all"
]


# ===============
# PL Trainer
# ===============
TRAIN_COMMON_PARAMS["trainer.num_epochs"] = 30
TRAIN_COMMON_PARAMS["trainer.num_devices"] = NUM_GPUS
TRAIN_COMMON_PARAMS["trainer.accelerator"] = "gpu"

# ===============
# Optimizer
# ===============
TRAIN_COMMON_PARAMS["opt.lr"] = 1e-5
TRAIN_COMMON_PARAMS["opt.weight_decay"] = 1e-3

# ===============
# Model
# ===============
if model_type == "CNN":
    TRAIN_COMMON_PARAMS["model"] = dict(
        dropout_rate=0.5,
        layers_description=(256,),
        tabular_data_inputs=[("data.input.clinical.all", 19)]
        if multimodality
        else None,
        tabular_layers_description=(128,) if multimodality else tuple(),
    )
elif model_type == "Transformer":
    token_dim = 768
    TRAIN_COMMON_PARAMS["model"] = dict(
        token_dim=token_dim,
        projection_kwargs=dict(
            image_shape=[300, 300], patch_shape=[30, 30], channels=3
        ),
        transformer_kwargs=dict(
            depth=12,
            heads=12,
            mlp_dim=token_dim * 4,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
        ),
    )

TRAIN_COMMON_PARAMS["model_type"] = model_type


def perform_softmax(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cls_preds = F.softmax(logits, dim=1)
    return logits, cls_preds


class MMViT(ViT):
    def __init__(
        self,
        token_dim: int,
        projection_kwargs: dict,
        transformer_kwargs: dict,
        multimodality: bool,
    ):
        super().__init__(token_dim, projection_kwargs, transformer_kwargs)
        self.multimodality = multimodality
        num_tokens = self.projection_layer.num_tokens
        self.token_dim = token_dim
        self._head = nn.Linear(token_dim, 8)
        if self.multimodality:
            # change pos embedding to accept additional token for multimodal
            self.transformer.pos_embedding = nn.Parameter(
                torch.randn(1, num_tokens + 2, token_dim)
            )

    # This forward can be Multimodal or just Imaging
    def forward(
        self, img_x: torch.Tensor, clinical_x: torch.Tensor = None
    ) -> torch.Tensor:
        img_x = self.projection_layer(img_x)
        if self.multimodality:
            clinical_x = clinical_x.unsqueeze(1)
            clinical_x_zeros = torch.zeros((img_x.shape[0], 1, self.token_dim))
            clinical_x_zeros[:, :, :19] = clinical_x
            clinical_x = clinical_x_zeros.cuda()
            x = torch.cat((img_x, clinical_x), 1)
        else:
            x = img_x
        x = self.transformer(x)
        x = self._head(x[:, 0])
        return x


def create_transformer_model(
    token_dim: int,
    projection_kwargs: dict,
    transformer_kwargs: dict,
) -> ModelWrapSeqToDict:
    torch_model = MMViT(
        token_dim=token_dim,
        projection_kwargs=projection_kwargs,
        transformer_kwargs=transformer_kwargs,
        multimodality=multimodality,
    )
    model = ModelWrapSeqToDict(
        model=torch_model,
        model_inputs=["data.input.img", "data.input.clinical.all"]
        if multimodality
        else ["data.input.img"],
        post_forward_processing_function=perform_softmax,
        model_outputs=["model.logits.head_0", "model.output.head_0"],
    )
    return model


def create_cnn_model(
    dropout_rate: float,
    layers_description: Sequence[int],
    tabular_data_inputs: Sequence[tuple[str, int]],
    tabular_layers_description: Sequence[int],
    backbone_type: str = "Resnet18",
) -> torch.nn.Module:
    """
    Creates the model

    :param backbone_type: (str) "InceptionResnetV2" or "Resnet18"
    """
    if backbone_type == "Resnet18":
        backbone = BackboneResnet(
            weights=ResNet18_Weights.IMAGENET1K_V1, in_channels=3, name="resnet18"
        )
        header_conv_inputs = [("model.backbone_features", 512)]
    elif backbone_type == "InceptionResnetV2":
        backbone = BackboneInceptionResnetV2(input_channels_num=3, logical_units_num=43)
        header_conv_inputs = [("model.backbone_features", 1536)]
    else:
        raise Exception(f"backbone_type ({backbone_type}) not supported")

    model = ModelMultiHead(
        conv_inputs=(("data.input.img", 3),),
        backbone=backbone,
        heads=[
            HeadGlobalPoolingClassifier(
                head_name="head_0",
                dropout_rate=dropout_rate,
                conv_inputs=header_conv_inputs,
                tabular_data_inputs=tabular_data_inputs,
                layers_description=layers_description,
                tabular_layers_description=tabular_layers_description,
                num_classes=8,
                pooling="avg",
            ),
        ],
    )
    return model


def create_datamodule(paths: dict, train_common_params: dict) -> pl.LightningDataModule:
    """
    In order to support the DDP strategy one need to create a Lightning Data Module.
    """
    datamodule = ISICDataModule(
        data_dir=paths["data_dir"],
        cache_dir=paths["cache_dir"],
        num_workers=train_common_params["data.num_workers"],
        batch_size=train_common_params["data.batch_size"],
        train_folds=train_common_params["data.train_folds"],
        validation_folds=train_common_params["data.validation_folds"],
        infer_folds=train_common_params["data.infer_folds"],
        split_filename=paths["data_split_filename"],
        sample_ids=train_common_params["data.samples_ids"],
        reset_cache=False,
        reset_split=False,
        use_batch_sampler=True if NUM_GPUS <= 1 else False,
        verify_folds_total_size=True,
    )

    return datamodule


#################################
# Train Template
#################################
def run_train(paths: dict, train_common_params: dict) -> None:
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(
        output_path=paths["model_dir"], console_verbose_level=logging.INFO
    )
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Train", {"attrs": ["bold", "underline"]})

    lgr.info(f'model_dir={paths["model_dir"]}', {"color": "magenta"})
    lgr.info(f'data_dir={paths["data_dir"]}', {"color": "magenta"})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {"color": "magenta"})

    # ==============================================================================
    # Data
    # ==============================================================================
    lgr.info("Datamodule:", {"attrs": "bold"})

    datamodule = create_datamodule(paths, train_common_params)

    lgr.info("Datamodule: Done", {"attrs": "bold"})

    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info("Model:", {"attrs": "bold"})
    model_type = train_common_params["model_type"]
    if model_type == "Transformer":
        model = create_transformer_model(**train_common_params["model"])
    elif model_type == "CNN":
        model = create_cnn_model(**train_common_params["model"])

    lgr.info("Model: Done", {"attrs": "bold"})

    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        "cls_loss": LossDefault(
            pred="model.logits.head_0",
            target="data.label",
            callable=F.cross_entropy,
            weight=1.0,
        ),
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    train_metrics = OrderedDict(
        [
            (
                "op",
                MetricApplyThresholds(pred="model.output.head_0"),
            ),  # will apply argmax
            (
                "auc",
                MetricAUCROC(
                    pred="model.output.head_0",
                    target="data.label",
                    class_names=class_names,
                ),
            ),
            (
                "accuracy",
                MetricAccuracy(pred="results:metrics.op.cls_pred", target="data.label"),
            ),
        ]
    )

    validation_metrics = copy.deepcopy(
        train_metrics
    )  # use the same metrics in validation as well

    best_epoch_source = dict(monitor="validation.metrics.auc.macro_avg", mode="max")

    # create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_common_params["opt.lr"],
        weight_decay=train_common_params["opt.weight_decay"],
    )

    # create learning scheduler
    lr_scheduler = {
        "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        "CosineAnnealing": optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1
        ),
    }["ReduceLROnPlateau"]
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")

    # optimizer and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    # =====================================================================================
    #  Train
    # =====================================================================================
    lgr.info("Train:", {"attrs": "bold"})

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

    # create lightning trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train_common_params["trainer.num_epochs"],
        accelerator=train_common_params["trainer.accelerator"],
        devices=train_common_params["trainer.num_devices"],
    )

    # train
    pl_trainer.fit(pl_module, datamodule=datamodule)

    lgr.info("Train: Done", {"attrs": "bold"})


######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS["infer_filename"] = "infer_file.gz"
INFER_COMMON_PARAMS["checkpoint"] = "best_epoch.ckpt"
INFER_COMMON_PARAMS["data.num_workers"] = NUM_WORKERS
INFER_COMMON_PARAMS["data.infer_folds"] = [4]  # infer validation set
INFER_COMMON_PARAMS["data.batch_size"] = 4

INFER_COMMON_PARAMS["model"] = TRAIN_COMMON_PARAMS["model"]
INFER_COMMON_PARAMS["trainer.num_devices"] = 1  # No need for multi-gpu in inference
INFER_COMMON_PARAMS["trainer.accelerator"] = TRAIN_COMMON_PARAMS["trainer.accelerator"]
INFER_COMMON_PARAMS["model_type"] = TRAIN_COMMON_PARAMS["model_type"]

######################################
# Inference Template
######################################


@rank_zero_only
def run_infer(paths: dict, infer_common_params: dict) -> None:
    create_dir(paths["inference_dir"])
    infer_file = os.path.join(
        paths["inference_dir"], infer_common_params["infer_filename"]
    )
    checkpoint_file = os.path.join(
        paths["model_dir"], infer_common_params["checkpoint"]
    )

    ## Logger
    fuse_logger_start(
        output_path=paths["inference_dir"], console_verbose_level=logging.INFO
    )
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Inference", {"attrs": ["bold", "underline"]})
    lgr.info(f"infer_filename={infer_file}", {"color": "magenta"})

    ## Data
    folds = load_pickle(
        paths["data_split_filename"]
    )  # assume exists and created in train func

    infer_sample_ids = []
    for fold in infer_common_params["data.infer_folds"]:
        infer_sample_ids += folds[fold]

    # Create dataset
    infer_dataset = ISIC.dataset(
        paths["data_dir"], paths["cache_dir"], samples_ids=infer_sample_ids, train=False
    )

    # dataloader
    infer_dataloader = DataLoader(
        dataset=infer_dataset,
        collate_fn=CollateDefault(),
        batch_size=infer_common_params["data.batch_size"],
        num_workers=infer_common_params["data.num_workers"],
    )

    # load python lightning module
    model_type = infer_common_params["model_type"]
    if model_type == "Transformer":
        model = create_transformer_model(**infer_common_params["model"])
    elif model_type == "CNN":
        model = create_cnn_model(**infer_common_params["model"])

    pl_module = LightningModuleDefault.load_from_checkpoint(
        checkpoint_file,
        model_dir=paths["model_dir"],
        model=model,
        map_location="cpu",
        strict=True,
    )
    # set the prediction keys to extract (the ones used be the evaluation function).
    pl_module.set_predictions_keys(
        ["model.output.head_0", "data.label"]
    )  # which keys to extract and dump into file

    # create a trainer instance
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        accelerator=infer_common_params["trainer.accelerator"],
        devices=infer_common_params["trainer.num_devices"],
        max_epochs=0,
    )
    predictions = pl_trainer.predict(
        pl_module, infer_dataloader, return_predictions=True
    )

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)


######################################
# Eval Common Params
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS["infer_filename"] = INFER_COMMON_PARAMS["infer_filename"]


######################################
# Eval Template
######################################
@rank_zero_only
def run_eval(paths: dict, eval_common_params: dict) -> None:
    infer_file = os.path.join(
        paths["inference_dir"], eval_common_params["infer_filename"]
    )

    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Eval", {"attrs": ["bold", "underline"]})

    # metrics
    metrics = OrderedDict(
        [
            (
                "op",
                MetricApplyThresholds(pred="model.output.head_0"),
            ),  # will apply argmax
            ("auc", MetricAUCROC(pred="model.output.head_0", target="data.label")),
            (
                "accuracy",
                MetricAccuracy(pred="results:metrics.op.cls_pred", target="data.label"),
            ),
            (
                "roc",
                MetricROCCurve(
                    pred="model.output.head_0",
                    target="data.label",
                    output_filename=os.path.join(
                        paths["inference_dir"], "roc_curve.png"
                    ),
                ),
            ),
        ]
    )

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(
        ids=None, data=infer_file, metrics=metrics, output_dir=paths["eval_dir"]
    )

    return results


######################################
# Run
######################################
if __name__ == "__main__":
    ## allocate gpus
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    ISIC.download(data_path=PATHS["data_dir"])

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
