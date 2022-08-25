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
import gzip
import pickle
import pandas as pd
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

from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC, MetricAccuracy
from fuseimg.datasets.cmmd import CMMD
from fuse.dl.models.backbones.backbone_inception_resnet_v2 import BackboneInceptionResnetV2
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe
from pytorch_lightning import Trainer
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.utils.file_io.file_io import create_dir, load_pickle, save_dataframe
from fuse.eval.evaluator import EvaluatorDefault
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
# assert "CMMD_DATA_PATH" in os.environ, "Expecting environment variable CMMD_DATA_PATH to be set. Follow the instruction in example README file to download and set the path to the data"
##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)
ROOT_PATH = 'vb'
CMMD_DATA_PATH = '/projects/msieve3/CMMD' # TODO: remove and provide instructions how to download the data 

def update_parameters_by_task(parameters=None,task_num=1):
    if parameters is None:
        parameters={}
        parameters['paths']={}
        parameters['train']={}
        parameters['infer']={}

    parameters['paths']['cache_dir'] =  os.path.join(ROOT_PATH, 'task_1_cache')
    if task_num == 1:
        parameters['task_num'] = 1
        parameters['paths']['model_dir'] =  os.path.join(ROOT_PATH, 'task_1_model_example')
        parameters['train']['target']= 'classification'
        parameters['train']['resume_checkpoint_filename'] = None
        parameters['train']['init_backbone'] = False
        parameters['train']['skip_keys'] = ['data.gt.subtype']

    elif task_num == 2:
        parameters['task_num'] = 1
        parameters['paths']['model_dir'] =  os.path.join(ROOT_PATH,  'task_2_model_example')
        parameters['train']['target']= 'subtype'
        parameters['train']['resume_checkpoint_filename'] = os.path.join(ROOT_PATH, 'task_1_model_example/checkpoint_best_0_epoch.pth')
        parameters['train']['init_backbone'] = True
        parameters['train']['skip_keys'] = []
        parameters['paths']['inference_dir'] = os.path.join(ROOT_PATH,'task_2_model_example/infer')
    if not os.path.isdir(ROOT_PATH):
        os.mkdir(ROOT_PATH)
    if not os.path.isdir(parameters['paths']['model_dir']):
        os.mkdir(parameters['paths']['model_dir'])
    return parameters
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
    elif train["target"] == "subtype":
        num_classes = 4
        gt_label = "data.gt.subtype"
        skip_keys = ["data.gt.classification"]
        class_names = ["Luminal A", "Luminal B", "HER2-enriched", "triple negative"]
    else:
        raise ("unsuported target!!")
    model = ModelMultiHead(
        conv_inputs=(("data.input.img", 1),),
        backbone=BackboneInceptionResnetV2(input_channels_num=1),
        heads=[
            HeadGlobalPoolingClassifier(
                head_name="head_0",
                dropout_rate=0.5,
                conv_inputs=[("model.backbone_features", 384)],
                layers_description=(256,),
                num_classes=num_classes,
                pooling="avg",
            ),
        ],
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
    return model, pl_trainer, num_classes, gt_label, skip_keys, class_names
#################################
# Train Template
#################################
def run_train(paths: NDict, train: NDict):

    model, pl_trainer, num_classes, gt_label, skip_keys, class_names = create_model(train, paths)
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')

    # Download data
    # TBD

    lgr.info('\nFuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})

    # split to folds randomly - temp
    dataset_all = CMMD.dataset(paths["data_dir"], paths["data_misc_dir"], train['target'], paths["cache_dir"],
                               reset_cache=False, num_workers=train["num_workers"], train=True)
    folds = dataset_balanced_division_to_folds(dataset=dataset_all,
                                               output_split_filename=os.path.join(paths["data_misc_dir"],
                                                                                  paths["data_split_filename"]),
                                               id='data.patientID',
                                               keys_to_balance=[gt_label],
                                               nfolds=train["num_folds"],
                                               workers=train["num_workers"])

    train_sample_ids = []
    for fold in train["train_folds"]:
        train_sample_ids += folds[fold]
    validation_sample_ids = []
    for fold in train["validation_folds"]:
        validation_sample_ids += folds[fold]

    train_dataset = CMMD.dataset(paths["data_dir"], paths["data_misc_dir"], train['target'], paths["cache_dir"],
                                 reset_cache=False, num_workers=train["num_workers"], sample_ids=train_sample_ids,
                                 train=True)

    validation_dataset = CMMD.dataset(paths["data_dir"], paths["data_misc_dir"], train['target'], paths["cache_dir"],
                                      reset_cache=False, num_workers=train["num_workers"],
                                      sample_ids=validation_sample_ids)

    ## Create sampler
    lgr.info(f'- Create sampler:')
    sampler = BatchSamplerDefault(dataset=train_dataset,
                                  balanced_class_name=gt_label,
                                  num_balanced_classes=num_classes,
                                  batch_size=train["batch_size"],
                                  mode=mode,
                                  balanced_class_weights=[1.0 / num_classes] * num_classes if train['target'] == "subtype" else None, )


    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=False, drop_last=False,
                                  batch_sampler=sampler, collate_fn=CollateDefault(skip_keys=train['skip_keys']),
                                  num_workers=train["num_workers"])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})

    #### Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})

    ## Create dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       shuffle=False,
                                       drop_last=False,
                                       batch_sampler=None,
                                       batch_size=train["batch_size"],
                                       num_workers=train["num_workers"],
                                       collate_fn=CollateDefault(skip_keys=train['skip_keys']))
    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})
    # ===================================================================
    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info('Model:', {'attrs': 'bold'})

    # load weights backbone weights
    # #load backbone weights
    if train['init_backbone']:
        checkpoint_file = train['resume_checkpoint_filename']
        # load python lightning module
        pl_module = LightningModuleDefault.load_from_checkpoint(
            checkpoint_file, model_dir=paths["model_dir"], model=model, map_location="cpu", strict=True
        )
        # set the prediction keys to extract (the ones used be the evaluation function).
        pl_module.set_predictions_keys(
            ["model.output.head_0", "data.gt.classification"]
        )  # which keys to extract and dump into file
        lgr.info("Test Data: Done", {"attrs": "bold"})

    lgr.info('Model: Done', {'attrs': 'bold'})

    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        "cls_loss": LossDefault(pred="model.logits.head_0", target=gt_label, callable=F.cross_entropy, weight=1.0)
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    train_metrics = OrderedDict(
        [
            ("op", MetricApplyThresholds(pred="model.output.head_0")),  # will apply argmax
            ("auc", MetricAUCROC(pred="model.output.head_0", target=gt_label, class_names=class_names)),
            ("accuracy", MetricAccuracy(pred="results:metrics.op.cls_pred", target=gt_label)),
        ]
    )

    validation_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
    best_epoch_source = dict(
        monitor="validation.metrics.auc.macro_avg",
        mode="max",
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

@hydra.main(config_path="conf", config_name="config")
def train_example(cfg: DictConfig) -> None:
    cfg = NDict(OmegaConf.to_container(cfg))
    cfg["paths"]["data_dir"] = CMMD_DATA_PATH

    task_loop = [1,2]
    for task_num in task_loop:
        cfg = update_parameters_by_task(cfg,task_num)
        print(cfg)
        run_train(cfg["paths"], cfg["train"])

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

    model, pl_trainer, num_classes, gt_label, skip_keys, class_names = create_model(train, paths)
    lgr.info("Model: Done", {"attrs": "bold"})
    ## Data
    folds = load_pickle(
        os.path.join(paths["data_misc_dir"], paths["data_split_filename"])
    )  # assume exists and created in train func

    infer_sample_ids = []
    for fold in infer["infer_folds"]:
        infer_sample_ids += folds[fold]

    test_dataset = CMMD.dataset(
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

    # create a trainer instance
    predictions = pl_trainer.predict(pl_module, infer_dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)

@hydra.main(config_path="conf", config_name="config")
def infer_example(cfg: DictConfig):
    cfg = NDict(OmegaConf.to_container(cfg))
    cfg["paths"]["data_dir"] = CMMD_DATA_PATH
    force_gpus = None  # [0]
    choose_and_enable_multiple_gpus(cfg["train.trainer.devices"], force_gpus=force_gpus)
    cfg = update_parameters_by_task(cfg, 2)

    infer_folds = [cfg['train']['train_folds'],cfg['train']['validation_folds'],cfg["infer"]["infer_folds"]]
    for infer_inx,infer_mode in enumerate(['train','validation','test']):
        cfg["infer"]["infer_filename"] = infer_mode+'_set_infer.gz'
        cfg["infer"]["infer_folds"] = infer_folds[infer_inx]
        # uncomment if you want to use specific gpus instead of automatically looking for free ones
        force_gpus = None  # [0]
        choose_and_enable_multiple_gpus(cfg["train.manager_train_params.num_gpus"], force_gpus=force_gpus)
        run_infer(cfg["paths"],cfg["infer"])

def save_results_csv(infer_file,csv_path=None,files_combined_path=None):
    if files_combined_path is None:
        files_combined_path = ''
    if csv_path is None:
        csv_path = os.path.join(infer_path,'infer_validation.csv')
    with gzip.open(infer_file, 'rb') as pickle_file:
        results_df = pickle.load(pickle_file)
    files_combined = pd.read_csv(files_combined_path)
    results_df[['pred_classA', 'pred_classB', 'pred_classC','pred_classD','pred_classE']] =\
        pd.DataFrame(results_df['model.output.head_0'].tolist(), index=results_df.index)
    results_df.drop('model.output.head_0', axis=1, inplace=True)
    merged = pd.merge(files_combined,results_df,left_on='file',right_on='id')
    merged.to_csv(csv_path)



if __name__ == "__main__":
    ##########################################
    # CNN training
    # 1) pretrain model using binary task
    # 2) Initialize weights and re-train on multi-class task
    ##########################################

    train_example()

    ##########################################
    # Infer and create csv file of prediction per class
    ##########################################


    infer_example()
    parameters = update_parameters_by_task(None,2)
    infer_path = parameters['paths']['inference_dir']
    for infer_inx,infer_mode in enumerate(['train','validation','test']):
        save_results_csv(os.path.join(infer_path,infer_mode+'_set_infer.gz'),
                         os.path.join(infer_path,infer_mode+'_set_infer.csv'),
                         os.path.join('/projects/msieve_dev3/usr/Tal/my_research/baseline_cmmd/cmmd_for_virtual_biopsy/data_misc','files_combined.csv'))