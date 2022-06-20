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
import gzip
import pickle
import pandas as pd
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
from fuse.dl.models.model_default import ModelDefault
from fuse.dl.models.heads.head_global_pooling_classifier import HeadGlobalPoolingClassifier
from fuse.utils.file_io.file_io import load_pickle
from fuse.dl.losses.loss_default import LossDefault

from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC, MetricAccuracy, MetricROCCurve

from fuse.dl.managers.callbacks.callback_tensorboard import TensorboardCallback
from fuse.dl.managers.callbacks.callback_metric_statistics import MetricStatisticsCallback
from fuse.dl.managers.callbacks.callback_time_statistics import TimeStatisticsCallback
from fuse.dl.managers.manager_default import ManagerDefault
from fuseimg.datasets.cmmd import CMMD
from fuse.dl.models.backbones.backbone_inception_resnet_v2 import BackboneInceptionResnetV2

import hydra
from typing import Dict
from omegaconf import DictConfig, OmegaConf
from fuse.eval.evaluator import EvaluatorDefault
from fuse.utils.dl.checkpoint import Checkpoint as FuseCheckpoint

# assert "CMMD_DATA_PATH" in os.environ, "Expecting environment variable CMMD_DATA_PATH to be set. Follow the instruction in example README file to download and set the path to the data"
##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

def update_parameters_by_task(parameters,task_num,root_path):
    parameters['paths']['cache_dir'] = root_path + 'task_1_cache'
    if task_num == 1:
        parameters['task_num'] = 1
        parameters['paths']['model_dir'] = root_path + '/task_1_model_example/'
        parameters['train']['target']= 'classification'
        parameters['train']['resume_checkpoint_filename'] = None
        parameters['train']['init_backbone'] = False
        parameters['train']['skip_keys'] = ['data.gt.subtype']


    elif task_num == 2:
        parameters['task_num'] = 1
        parameters['paths']['model_dir'] = root_path + '/task_2_model_example/'
        parameters['train']['target']= 'subtype'
        parameters['train']['resume_checkpoint_filename'] = root_path + '/task_1_model_example/checkpoint_best_0_epoch.pth'
        parameters['train']['init_backbone'] = True
        parameters['train']['skip_keys'] = []
        parameters['infer']['inference_dir'] = root_path +'infer/'



    return parameters

#################################
# Train Template
#################################
def run_train(paths: NDict, train: NDict):

    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    choose_and_enable_multiple_gpus(train['manager_train_params']["num_gpus"], force_gpus=force_gpus)

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

    #### Train Data
    if train['target'] == "classification":
        num_classes = 2
        mode = "approx"
        gt_label = "data.gt.classification"
        class_names = ["Benign", "Malignant"]
    elif train['target'] == "subtype":
        num_classes = 5
        mode = "approx"
        gt_label = "data.gt.subtype"
        class_names = ["Luminal A", "Luminal B", "HER2-enriched", "triple negative","else"]
    else:
        raise ("unsuported target!!")
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

    model = ModelDefault(
        conv_inputs=(('data.input.img', 1),),
        backbone=BackboneInceptionResnetV2(input_channels_num=1),
        heads=[
            HeadGlobalPoolingClassifier(head_name='head_0',
                                        dropout_rate=0.5,
                                        conv_inputs=[('model.backbone_features', 384)],
                                        layers_description=(256,),
                                        num_classes=num_classes,
                                        pooling="avg"),
        ]
    )

    # load weights backbone weights
    # #load backbone weights
    if train['init_backbone']:
        checkpoint_file = train['resume_checkpoint_filename']
        net_state_dict_list = FuseCheckpoint.load_from_file(checkpoint_file)
        backbone_state = model.backbone.state_dict()
        state_dict = net_state_dict_list.net_state_dict
        for name, param in state_dict.items():
            print(name)
            if ('backbone' in name):
                target_name = name.split('backbone.')[1]
                backbone_state[target_name].copy_(param.data)

        model.backbone.load_state_dict(*[backbone_state], strict=True)
        train['resume_checkpoint_filename'] = None

    lgr.info('Model: Done', {'attrs': 'bold'})

    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        'cls_loss': LossDefault(pred='model.logits.head_0', target=gt_label,
                                callable=F.cross_entropy, weight=1.0)
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    metrics = OrderedDict([
        ('op', MetricApplyThresholds(pred='model.output.head_0')),  # will apply argmax
        ('auc', MetricAUCROC(pred='model.output.head_0', target=gt_label, class_names=class_names)),
        ('accuracy', MetricAccuracy(pred='results:metrics.op.cls_pred', target=gt_label)),
    ])

    # =====================================================================================
    #  Callbacks
    # =====================================================================================
    callbacks = [
        # default callbacks
        TensorboardCallback(model_dir=paths['model_dir']),  # save statistics for tensorboard
        MetricStatisticsCallback(output_path=paths['model_dir'] + "/metrics.csv"),  # save statistics in a csv file
        TimeStatisticsCallback(num_epochs=train["manager_train_params"]["num_epochs"], load_expected_part=0.1)
        # time profiler
    ]

    # =====================================================================================
    #  Manager - Train
    #  Create a manager, training objects and run a training process.
    # =====================================================================================
    lgr.info('Train:', {'attrs': 'bold'})

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=train["learning_rate"],
                           weight_decay=train["weight_decay"])

    # create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # train from scratch

    manager = ManagerDefault(output_model_dir=paths['model_dir'], force_reset=train['force_reset_model_dir'])

    # Providing the objects required for the training process.
    manager.set_objects(net=model,
                        optimizer=optimizer,
                        losses=losses,
                        metrics=metrics,
                        best_epoch_source=train["manager_best_epoch_source"],
                        lr_scheduler=scheduler,
                        callbacks=callbacks,
                        train_params=train["manager_train_params"],
                        output_model_dir=paths["model_dir"])

    # Continue training
    if train["resume_checkpoint_filename"] is not None:
        # Loading the checkpoint including model weights, learning rate, and epoch_index.
        manager.load_checkpoint(checkpoint=train["resume_checkpoint_filename"], mode='train',
                                values_to_resume=['net'])
    # # Start training
    manager.train(train_dataloader=train_dataloader,
                  validation_dataloader=validation_dataloader)

    lgr.info('Train: Done', {'attrs': 'bold'})

@hydra.main(config_path="conf", config_name="config")
def train_example(cfg: DictConfig) -> None:
    root_path = '/projects/msieve_dev3/usr/Tal/my_research/baseline_cmmd/cmmd_for_virtual_biopsy/'
    cfg = NDict(OmegaConf.to_container(cfg))
    cfg["paths"]["data_dir"] = '/projects/msieve3/CMMD'
    # os.environ["CMMD_DATA_PATH"]
    task_loop = [2]
    for task_num in task_loop:
        cfg = update_parameters_by_task(cfg,task_num,root_path)
        print(cfg)
        run_train(cfg["paths"], cfg["train"])

######################################
# Inference Template
######################################
def run_infer(paths: NDict, infer: NDict):
    #### Logger
    fuse_logger_start(output_path=paths["inference_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={os.path.join(paths["inference_dir"], infer["infer_filename"])}', {'color': 'magenta'})

    ## Data
    folds = load_pickle(paths["data_split_filename"])  # assume exists and created in train func

    infer_sample_ids = []
    for fold in infer["infer_folds"]:
        infer_sample_ids += folds[fold]

    test_dataset = CMMD.dataset(paths["data_dir"], paths["data_misc_dir"], infer['target'], paths["cache_dir"],
                                sample_ids=infer_sample_ids, train=False)

    ## Create dataloader
    infer_dataloader = DataLoader(dataset=test_dataset,
                                  shuffle=False, drop_last=False,
                                  collate_fn=CollateDefault(),
                                  num_workers=infer["num_workers"])
    lgr.info(f'Test Data: Done', {'attrs': 'bold'})

    #### Manager for inference
    manager = ManagerDefault()
    # extract just the global classification per sample and save to a file
    output_columns = ['model.output.head_0', 'data.gt.classification']
    manager.infer(data_loader=infer_dataloader,
                  input_model_dir=paths["model_dir"],
                  checkpoint=infer["checkpoint"],
                  output_columns=output_columns,
                  output_file_name=os.path.join(paths["inference_dir"], infer["infer_filename"]))

@hydra.main(config_path="conf", config_name="config")
def infer_example(cfg: DictConfig):
    cfg = NDict(OmegaConf.to_container(cfg))
    cfg["paths"]["data_dir"] = '/projects/msieve3/CMMD'
    cfg = update_parameters_by_task(cfg, 2 , root_path)

    infer_folds = [cfg['train']['train_folds'],cfg['train']['validation_folds'],cfg["infer"]["infer_folds"]]
    for infer_inx,infer_mode in enumerate(['train','validation','test']):
        cfg["infer"]["infer_filename"] = infer_mode+'_set_infer.gz'
        cfg["infer"]["infer_folds"] = infer_folds[infer_inx]
        # uncomment if you want to use specific gpus instead of automatically looking for free ones
        force_gpus = None  # [0]
        choose_and_enable_multiple_gpus(cfg["train.manager_train_params.num_gpus"], force_gpus=force_gpus)
        run_infer(cfg["paths"],cfg["infer"])

def save_results_csv(infer_file,csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(infer_path,'infer_validation.csv')
    with gzip.open(infer_file, 'rb') as pickle_file:
        results_df = pickle.load(pickle_file)
    results_df[['pred_classA', 'pred_classB', 'pred_classC','pred_classD','pred_classE']] =\
        pd.DataFrame(results_df['model.output.head_0'].tolist(), index=results_df.index)
    results_df.drop('model.output.head_0', axis=1, inplace=True)
    results_df.to_csv(csv_path)



if __name__ == "__main__":
    sys.argv.append('hydra.run.dir=working_dir')

    ##########################################
    # CNN training
    # 1) pretrain model using binary task
    # 2) Initialize weights and re-train on multi-class task
    ##########################################
    data_path = '/projects/msieve3/CMMD'
    root_path = '/projects/msieve_dev3/usr/Tal/my_research/baseline_cmmd/cmmd_for_virtual_biopsy/'

    parameters = train_example()
    #
    # # ##########################################
    # # # Infer and create csv file of prediction per class
    # # ##########################################
    infer_path =root_path+'infer/'

    infer_example()
    for infer_inx,infer_mode in enumerate(['train','validation','test']):
        save_results_csv(infer_path+infer_mode+'_set_infer.pickle.gz',infer_mode+'_set_infer.csv')
