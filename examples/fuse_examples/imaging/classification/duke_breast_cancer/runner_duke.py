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

import logging
import os
#todo: remove setting of env variable !!!
os.environ["DUKE_DATA_PATH"] = "/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/"
import getpass
from typing import OrderedDict
from fuse.utils.file_io.file_io import load_pickle


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from fuse.eval.evaluator import EvaluatorDefault
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve

from fuse.eval.evaluator import EvaluatorDefault
from fuse.data.sampler.sampler_balanced_batch import FuseSamplerBalancedBatch
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.split import dataset_balanced_division_to_folds
    
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.managers.callbacks.callback_metric_statistics import MetricStatisticsCallback
from fuse.dl.managers.callbacks.callback_tensorboard import TensorboardCallback
from fuse.dl.managers.callbacks.callback_time_statistics import TimeStatisticsCallback
from fuse.dl.managers.manager_default import ManagerDefault
from fuse.dl.models.backbones.backbone_resnet_3d import BackboneResnet3D
from fuse.dl.models.model_default import ModelDefault
from fuse.dl.models.heads.head_3D_classifier import Head3dClassifier

from fuse.utils.utils_debug import FuseDebug
import fuse.utils.gpu as GPU
from fuse.utils.utils_logger import fuse_logger_start

from fuseimg.datasets import duke

from fuse.dl.models.heads.head_1d_classifier import Head1dClassifier

from examples.fuse_examples.imaging.classification.prostate_x.backbone_3d_multichannel import Fuse_model_3d_multichannel,ResNet
from examples.fuse_examples.imaging.classification.prostate_x.patient_data_source import ProstateXDataSourcePatient



from examples.fuse_examples.imaging.classification.duke_breast_cancer.dataset import duke_breast_cancer_dataset
from examples.fuse_examples.imaging.classification.duke_breast_cancer.tasks import Task


###########################################################################################################
# Fuse
###########################################################################################################
##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'fast', 'debug', 'verbose', 'user'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################
# Output Paths
##########################################
assert "DUKE_DATA_PATH" in os.environ, "Expecting environment variable DUKE_DATA_PATH to be set. Follow the instruction in example README file to download and set the path to the data"
ROOT = f'/projects/msieve_dev3/usr/{getpass.getuser()}/fuse_examples/duke'
model_dir = os.path.join(ROOT, 'model_dir')
PATHS = {'model_dir': model_dir,
         'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'cache_dir': os.path.join(ROOT, 'cache_dir'),
         'data_split_filename': os.path.join(ROOT, 'DUKE_folds_fuse2_11102021TumorSize_seed1.pkl'),
         'data_dir': os.environ["DUKE_DATA_PATH"],
         'inference_dir': os.path.join(model_dir, 'infer_dir'),
         'eval_dir': os.path.join(model_dir, 'eval_dir'),
         }  #todo: add annotations file

##########################################
# Train Common Params
##########################################
TRAIN_COMMON_PARAMS = {}
# ============
# Model
# ============

# ============
# Data
# ============
TRAIN_COMMON_PARAMS['data.batch_size'] = 4
TRAIN_COMMON_PARAMS['data.train_num_workers'] = 16
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 16
TRAIN_COMMON_PARAMS['data.num_folds'] = 5
TRAIN_COMMON_PARAMS['data.train_folds'] = [0, 1, 2]
TRAIN_COMMON_PARAMS['data.validation_folds'] = [3]


# ===============
# Manager - Train
# ===============
TRAIN_COMMON_PARAMS['manager.train_params'] = {
    'device': 'cuda', 
    'num_epochs': 5,
    'virtual_batch_size': 1,  # number of batches in one virtual batch
    'start_saving_epochs': 120,  # first epoch to start saving checkpoints from
    'gap_between_saving_epochs': 1,  # number of epochs between saved checkpoint
}
TRAIN_COMMON_PARAMS['manager.best_epoch_source'] = {
    'source': 'metrics.auc.macro_avg',  # can be any key from 'epoch_results'
    'optimization': 'max',  # can be either min/max
    'on_equal_values': 'better',
    # can be either better/worse - whether to consider best epoch when values are equal
}
TRAIN_COMMON_PARAMS['manager.learning_rate'] = 1e-5
TRAIN_COMMON_PARAMS['manager.weight_decay'] = 0.001
TRAIN_COMMON_PARAMS['manager.dropout'] = 0.5
TRAIN_COMMON_PARAMS['manager.momentum'] = 0.9
TRAIN_COMMON_PARAMS['manager.resume_checkpoint_filename'] = None  # if not None, will try to load the checkpoint
TRAIN_COMMON_PARAMS['imaging_dropout'] = 0.25
# TRAIN_COMMON_PARAMS['fused_dropout'] = 0.0
# TRAIN_COMMON_PARAMS['clinical_dropout'] = 0.0

TRAIN_COMMON_PARAMS['num_backbone_features_imaging'] = 512

# in order to add relevant tabular feature uncomment:
# num_backbone_features_clinical, post_concat_inputs,post_concat_model
TRAIN_COMMON_PARAMS['num_backbone_features_clinical'] = None#256
TRAIN_COMMON_PARAMS['post_concat_inputs'] = None#[('data.clinical_features',9),]
TRAIN_COMMON_PARAMS['post_concat_model'] = None#(256,256)

if TRAIN_COMMON_PARAMS['num_backbone_features_clinical'] is None:
    TRAIN_COMMON_PARAMS['num_backbone_features'] = TRAIN_COMMON_PARAMS['num_backbone_features_imaging']
else:
    TRAIN_COMMON_PARAMS['num_backbone_features'] = \
            TRAIN_COMMON_PARAMS['num_backbone_features_imaging']+TRAIN_COMMON_PARAMS['num_backbone_features_clinical']

# classification_task:
# supported tasks are: 'Staging Tumor Size','Histology Type','is High Tumor Grade Total','PCR'

TRAIN_COMMON_PARAMS['classification_task'] = 'Staging Tumor Size'
TRAIN_COMMON_PARAMS['task'] = Task(TRAIN_COMMON_PARAMS['classification_task'], 0)
TRAIN_COMMON_PARAMS['class_num'] = TRAIN_COMMON_PARAMS['task'].num_classes()

# backbone parameters
TRAIN_COMMON_PARAMS['backbone_model_dict'] = \
    {'input_channels_num': 1,
     }


#################################
# Train Template
#################################
def run_train(paths: dict, train_params: dict):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})

    # ==============================================================================
    # Data
    # ==============================================================================
    # Train Data
    lgr.info(f'Train Data:', {'attrs': 'bold'})

    # split to folds randomly - temp
    dataset_all = duke.Duke.dataset(paths["data_dir"], paths["cache_dir"], reset_cache=False)
    folds = dataset_balanced_division_to_folds(dataset=dataset_all,
                                        output_split_filename=paths["data_split_filename"], 
                                        keys_to_balance=["data.ground_truth"],
                                        nfolds=train_params["data.num_folds"])

    train_sample_ids = []
    for fold in train_params["data.train_folds"]:
        train_sample_ids += folds[fold]
    validation_sample_ids = []
    for fold in train_params["data.validation_folds"]:
        validation_sample_ids += folds[fold]

    train_dataset = duke.Duke.dataset(paths["data_dir"], paths["cache_dir"], sample_ids=train_sample_ids, train=True)
    # for _ in train_dataset:
    #     pass
    validation_dataset = duke.Duke.dataset(paths["data_dir"], paths["cache_dir"], sample_ids=validation_sample_ids, train=False)

    lgr.info(f'- Create sampler:')
    sampler = FuseSamplerBalancedBatch(dataset=train_dataset,
                                       balanced_class_name='data.ground_truth',
                                       num_balanced_classes=2,
                                       batch_size=train_params['data.batch_size'],
                                       balanced_class_weights=None,
                                       use_dataset_cache=True)
    lgr.info(f'- Create sampler: Done')

    # Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_sampler=sampler,
                                  collate_fn=train_dataset.collate_fn, #previously: CollateDefault(),
                                  num_workers=train_params['data.train_num_workers'])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})


    # dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       batch_size=train_params['data.batch_size'],
                                       collate_fn=validation_dataset.collate_fn, #CollateDefault(),
                                       num_workers=train_params['data.validation_num_workers'])
    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})

    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info('Model:', {'attrs': 'bold'})

    model = Fuse_model_3d_multichannel(
    conv_inputs=(('data.input', 1),),
    backbone=ResNet(ch_num=TRAIN_COMMON_PARAMS['backbone_model_dict']['input_channels_num']),
    # since backbone resnet contains pooling and fc, the feature output is 1D,
    # hence we use Head1dClassifier as classification head
    heads=[
        Head1dClassifier(head_name='isLargeTumorSize',
                         conv_inputs=[('model.backbone_features', train_params['num_backbone_features'])],
                         post_concat_inputs=train_params['post_concat_inputs'],
                         post_concat_model=train_params['post_concat_model'],
                         dropout_rate=train_params['imaging_dropout'],
                         # append_dropout_rate=train_params['clinical_dropout'],
                         # fused_dropout_rate=train_params['fused_dropout'],
                         shared_classifier_head=None,
                         layers_description=None,
                             num_classes=2,
                             # append_features=[("data.input.clinical", 8)],
                             # append_layers_description=(256,128),
                             ),
    ]
)


    lgr.info('Model: Done', {'attrs': 'bold'})

    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        'cls_loss': LossDefault(pred='model.logits.isLargeTumorSize',
                                target='data.ground_truth', callable=F.cross_entropy, weight=1.0),
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    lgr.info('Metrics:', {'attrs': 'bold'})
    metrics = OrderedDict([
        ('auc', MetricAUCROC(pred='model.output.isLargeTumorSize', target='data.ground_truth')),
    ])

    # =====================================================================================
    #  Callbacks
    # =====================================================================================
    callbacks = [
        # default callbacks
        TensorboardCallback(model_dir=paths['model_dir']),  # save statistics for tensorboard
        MetricStatisticsCallback(output_path=paths['model_dir'] + "/metrics.csv"),  # save statistics a csv file
        TimeStatisticsCallback(num_epochs=train_params['manager.train_params']['num_epochs'], load_expected_part=0.1)  # time profiler
    ]

    # =====================================================================================
    #  Manager - Train
    # =====================================================================================
    lgr.info('Train:', {'attrs': 'bold'})

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=train_params['manager.learning_rate'],
                           weight_decay=train_params['manager.weight_decay'])

    # create learning scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # train from scratch
    manager = ManagerDefault(output_model_dir=paths['model_dir'], force_reset=paths['force_reset_model_dir'])
    # Providing the objects required for the training process.
    manager.set_objects(net=model,
                        optimizer=optimizer,
                        losses=losses,
                        metrics=metrics,
                        best_epoch_source=train_params['manager.best_epoch_source'],
                        lr_scheduler=scheduler,
                        callbacks=callbacks,
                        train_params=train_params['manager.train_params'])

    ## Continue training
    if train_params['manager.resume_checkpoint_filename'] is not None:
        # Loading the checkpoint including model weights, learning rate, and epoch_index.
        manager.load_checkpoint(checkpoint=train_params['manager.resume_checkpoint_filename'], mode='train')

    # Start training
    manager.train(train_dataloader=train_dataloader, validation_dataloader=validation_dataloader)

    lgr.info('Train: Done', {'attrs': 'bold'})


######################################
# Inference Common Params
######################################
#todo: I'm here !!!
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS['infer_filename'] = 'validation_set_infer.gz'
INFER_COMMON_PARAMS['checkpoint'] = 'best'  # Fuse TIP: possible values are 'best', 'last' or epoch_index.
INFER_COMMON_PARAMS['data.infer_folds'] = [4]  # infer validation set
INFER_COMMON_PARAMS['data.batch_size'] = 4
INFER_COMMON_PARAMS['data.num_workers'] = 16


######################################
# Inference Template
######################################
def run_infer(paths: dict, infer_common_params: dict):
    #### Logger
    fuse_logger_start(output_path=paths['inference_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={os.path.join(paths["inference_dir"], infer_common_params["infer_filename"])}', {'color': 'magenta'})

    ## Data
    folds = load_pickle(paths["data_split_filename"]) # assume exists and created in train func

    infer_sample_ids = []                              
    for fold in infer_common_params["data.infer_folds"]:
        infer_sample_ids += folds[fold]

    validation_dataset = STOIC21.dataset(paths["data_dir"], paths["cache_dir"], sample_ids=infer_sample_ids, train=False)

    # dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=infer_common_params['data.batch_size'], collate_fn=CollateDefault(),
                                       num_workers=infer_common_params['data.num_workers'])


    ## Manager for inference
    manager = ManagerDefault()
    output_columns = ['model.output.classification', 'data.gt.probSevere']
    manager.infer(data_loader=validation_dataloader,
                  input_model_dir=paths['model_dir'],
                  checkpoint=infer_common_params['checkpoint'],
                  output_columns=output_columns,
                  output_file_name=os.path.join(paths["inference_dir"], infer_common_params["infer_filename"]))


######################################
# Analyze Common Params
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS['infer_filename'] = INFER_COMMON_PARAMS['infer_filename']


######################################
# Analyze Template
######################################
def run_eval(paths: dict, eval_common_params: dict):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Analyze', {'attrs': ['bold', 'underline']})

    # metrics
    metrics = OrderedDict([
        ('operation_point', MetricApplyThresholds(pred='model.output.classification')), # will apply argmax
        ('accuracy', MetricAccuracy(pred='results:metrics.operation_point.cls_pred', target='data.gt.probSevere')),
        ('roc', MetricROCCurve(pred='model.output.classification', target='data.gt.probSevere', output_filename=os.path.join(paths['inference_dir'], 'roc_curve.png'))),
        ('auc', MetricAUCROC(pred='model.output.classification', target='data.gt.probSevere')),
    ])
   
    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(ids=None,
                     data=os.path.join(paths["inference_dir"], eval_common_params["infer_filename"]),
                     metrics=metrics,
                     output_dir=paths['eval_dir'])

    return results


######################################
# Run
######################################
if __name__ == "__main__":
    # allocate gpus
    # To use cpu - set NUM_GPUS to 0
    NUM_GPUS = 1
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None # [0]
    GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    RUNNING_MODES = ['train', 'infer', 'eval']  # Options: 'train', 'infer', 'eval'
    # train
    if 'train' in RUNNING_MODES:
        run_train(paths=PATHS, train_params=TRAIN_COMMON_PARAMS)

    # infer
    if 'infer' in RUNNING_MODES:
        run_infer(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    # eval
    if 'eval' in RUNNING_MODES:
        run_eval(paths=PATHS, eval_common_params=EVAL_COMMON_PARAMS)

