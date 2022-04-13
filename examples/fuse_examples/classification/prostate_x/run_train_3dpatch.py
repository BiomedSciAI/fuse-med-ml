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
import pathlib
from fuse.data.dataset.dataset_base import FuseDatasetBase
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC, MetricROCCurve
from fuse.eval.evaluator import EvaluatorDefault
from fuse.data.dataset.dataset_base import FuseDatasetBase

from fuse.data.sampler.sampler_balanced_batch import FuseSamplerBalancedBatch
from fuse.losses.loss_default import FuseLossDefault
from fuse.managers.callbacks.callback_metric_statistics import FuseMetricStatisticsCallback
from fuse.managers.callbacks.callback_tensorboard import FuseTensorboardCallback
from fuse.managers.callbacks.callback_time_statistics import FuseTimeStatisticsCallback
from fuse.managers.manager_default import FuseManagerDefault

import fuse.utils.gpu as FuseUtilsGPU
from fuse.utils.utils_logger import fuse_logger_start


from fuse_examples.classification.prostate_x.dataset import prostate_x_dataset
from fuse_examples.classification.prostate_x.backbone_3d_multichannel import Fuse_model_3d_multichannel,ResNet
from fuse_examples.classification.prostate_x.patient_data_source import FuseProstateXDataSourcePatient
from fuse_examples.classification.prostate_x.tasks import FuseProstateXTask
from fuse.models.heads.head_1d_classifier import FuseHead1dClassifier


##########################################
# Output Paths
# ##########################################

# TODO: path to save model
root_path  ='.'
# TODO: path for prostateX data
# Download instructions can be found in README
# load data from:
# https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges#23691656d4622c5ad5884bdb876d6d441994da38
root_data = 'PatientData/ProstateX/manifest-A3Y4AE4o5818678569166032044/'


PATHS = {'force_reset_model_dir': False,
         # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'model_dir': os.path.join(root_path, 'prostatex/my_model/'),
         'cache_dir': os.path.join(root_path, 'prostatex/my_cache/'),
         'inference_dir': os.path.join(root_path, 'prostatex/my_model/inference/'),
         'eval_dir': os.path.join(root_path,  'prostatex/my_model/eval/'),
         'data_dir': pathlib.Path(__file__).parent.resolve(),
         'prostate_data_path' : root_data,
         'ktrans_path': os.path.join(root_data, 'ProstateXKtrains-train-fixed/'),
         }
#################################
# Train Template
#################################
##########################################
# Train Common Params
##########################################
# ============
# Data
# ============
TRAIN_COMMON_PARAMS = {}
TRAIN_COMMON_PARAMS['db_name'] = 'prostate_x'
TRAIN_COMMON_PARAMS['db_version'] = 29062021
TRAIN_COMMON_PARAMS['fold_no'] = 5
TRAIN_COMMON_PARAMS['data.batch_size'] = 50
TRAIN_COMMON_PARAMS['data.train_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 8
# add misalignment to segmentation
TRAIN_COMMON_PARAMS['data.aug.mask_misalignment'] = True
# add misalignment to phase registration
TRAIN_COMMON_PARAMS['data.aug.phase_misalignment'] = True

# ===============
# Manager - Train
# ===============
TRAIN_COMMON_PARAMS['manager.train_params'] = {
    'num_gpus': 1,
    'num_epochs': 5,
    'virtual_batch_size': 1,  # number of batches in one virtual batch
    'start_saving_epochs': 120,  # first epoch to start saving checkpoints from
    'gap_between_saving_epochs': 5,  # number of epochs between saved checkpoint
}
TRAIN_COMMON_PARAMS['manager.best_epoch_source'] = [
    {
        'source': 'metrics.auc.macro_avg',  # can be any key from losses or metrics dictionaries
        'optimization': 'max',  # can be either min/max
        'on_equal_values': 'better',
        # can be either better/worse - whether to consider best epoch when values are equal
    },

]
TRAIN_COMMON_PARAMS['manager.learning_rate'] = 1e-5
TRAIN_COMMON_PARAMS['manager.weight_decay'] = 1e-4
TRAIN_COMMON_PARAMS['manager.dropout'] = 0.5
TRAIN_COMMON_PARAMS['manager.momentum'] = 0.9
TRAIN_COMMON_PARAMS['manager.resume_checkpoint_filename'] = None  # if not None, will try to load the checkpoint

TRAIN_COMMON_PARAMS['num_backbone_features'] = 512
TRAIN_COMMON_PARAMS['task'] = FuseProstateXTask('ClinSig', 0)
TRAIN_COMMON_PARAMS['class_num'] = TRAIN_COMMON_PARAMS['task'].num_classes()

# backbone parameters
TRAIN_COMMON_PARAMS['backbone_model_dict'] = \
    {'input_channels_num': 5,
     }

def train_template(paths: dict, train_common_params: dict):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO,
                      list_of_source_files=[])
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})

    # ==============================================================================
    # Data
    # ==============================================================================
    train_dataset, validation_dataset = prostate_x_dataset(paths,train_common_params,lgr)

    ## Create dataloader
    lgr.info(f'- Create sampler:')

    sampler = FuseSamplerBalancedBatch(dataset=train_dataset,
                                       balanced_class_name='data.ground_truth',
                                       num_balanced_classes=train_common_params['task'].num_classes(),
                                       batch_size=train_common_params['data.batch_size'],
                                       balanced_class_weights=
                                       [int(train_common_params['data.batch_size']/train_common_params['class_num'])] * train_common_params['class_num'],
                                       use_dataset_cache=True)

    lgr.info(f'- Create sampler: Done')

    # ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_sampler=sampler,
                                  collate_fn=train_dataset.collate_fn,
                                  num_workers=train_common_params['data.train_num_workers'])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})

    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       shuffle=False,
                                       drop_last=False,
                                       batch_size=train_common_params['data.batch_size'],
                                       num_workers=train_common_params['data.validation_num_workers'],
                                       collate_fn=validation_dataset.collate_fn)
    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})

    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info('Model:', {'attrs': 'bold'})

    model = Fuse_model_3d_multichannel(
        conv_inputs=(('data.input', 1),),
        backbone= ResNet(ch_num=TRAIN_COMMON_PARAMS['backbone_model_dict']['input_channels_num']),
        heads=[
        FuseHead1dClassifier(head_name='ClinSig',
                                        conv_inputs=[('model.backbone_features',  train_common_params['num_backbone_features'])],
                                        post_concat_inputs=None,
                                        dropout_rate=0.25,
                                        shared_classifier_head=None,
                                        layers_description=None,
                                        num_classes=2),

        ]
    )
    lgr.info('Model: Done', {'attrs': 'bold'})

    # ====================================================================================
    #  Loss
    # ====================================================================================
    lgr.info('Losses: CrossEntropy', {'attrs': 'bold'})

    losses = {
            'cls_loss': FuseLossDefault(pred_name='model.logits.ClinSig',
                                        target_name='data.ground_truth',
                                        callable=F.cross_entropy, weight=1.0),
        }


    # ====================================================================================
    # Metrics
    # ====================================================================================
    lgr.info('Metrics:', {'attrs': 'bold'})

    metrics = {

        'auc': MetricAUCROC(pred='model.output.ClinSig', target='data.ground_truth',
                                class_names=train_common_params['task'].class_names()),
    }


    # =====================================================================================
    #  Callbacks
    # =====================================================================================
    callbacks = [
        FuseTensorboardCallback(model_dir=paths['model_dir']),  # save statistics for tensorboard
        FuseMetricStatisticsCallback(output_path=paths['model_dir'] + "/metrics.csv"),
        # save statistics for tensorboard in a csv file
        FuseTimeStatisticsCallback(num_epochs=train_common_params['manager.train_params']['num_epochs'],
                                   load_expected_part=0.1)  # time profiler
    ]

    # =====================================================================================
    #  Manager - Train
    # =====================================================================================
    lgr.info('Train:', {'attrs': 'bold'})

    # create optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=train_common_params['manager.learning_rate'],
                           weight_decay=train_common_params['manager.weight_decay'])

    # create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # train from scratch
    manager = FuseManagerDefault(output_model_dir=paths['model_dir'], force_reset=paths['force_reset_model_dir'])
    # Providing the objects required for the training process.
    manager.set_objects(net=model,
                        optimizer=optimizer,
                        losses=losses,
                        metrics=metrics,
                        best_epoch_source=train_common_params['manager.best_epoch_source'],
                        lr_scheduler=scheduler,
                        callbacks=callbacks,
                        train_params=train_common_params['manager.train_params'])

    ## Continue training
    if train_common_params['manager.resume_checkpoint_filename'] is not None:
        # Loading the checkpoint including model weights, learning rate, and epoch_index.
        manager.load_checkpoint(checkpoint=train_common_params['manager.resume_checkpoint_filename'], mode='train')

    # Start training
    manager.train(train_dataloader=train_dataloader,
                  validation_dataloader=validation_dataloader)

    lgr.info('Train: Done', {'attrs': 'bold'})



######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS['db_version'] = 29062021
INFER_COMMON_PARAMS['db_name'] = 'prostate_x'
INFER_COMMON_PARAMS['fold_no'] = 5
INFER_COMMON_PARAMS['infer_filename'] = os.path.join(PATHS['inference_dir'], 'validation_set_infer.pickle.gz')
INFER_COMMON_PARAMS['checkpoint'] = 'best' # Fuse TIP: possible values are 'best', 'last' or epoch_index.

######################################
# Inference Template
######################################
def infer_template(paths: dict, infer_common_params: dict):
    #### Logger
    # fuse_logger_start(output_path=paths['inference_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={infer_common_params["infer_filename"]}', {'color': 'magenta'})

    #### create infer data set

    lgr.info(f'db_name={infer_common_params["db_name"]}', {'color': 'magenta'})
    ## Create data source:
    infer_data_source = FuseProstateXDataSourcePatient(paths['data_dir'],'validation',
                                                       db_ver=infer_common_params['db_version'],
                                                       db_name = infer_common_params['db_name'],
                                                       fold_no=infer_common_params['fold_no'])

    ### load dataset
    data_set_filename = os.path.join(paths["model_dir"], "inference_dataset.pth")
    dataset = FuseDatasetBase.load(filename=data_set_filename, override_datasource=infer_data_source, override_cache_dest=paths["cache_dir"], num_workers=0)
    dataloader  = DataLoader(dataset=dataset,
                                       shuffle=False,
                                       drop_last=False,
                                       batch_size=50,
                                       num_workers=5,
                                       collate_fn=dataset.collate_fn)
    #### Manager for inference
    manager = FuseManagerDefault()
    # extract just the global classification per sample and save to a file
    output_columns = ['model.output.ClinSig','data.ground_truth']
    manager.infer(data_loader=dataloader,
                  input_model_dir=paths['model_dir'],
                  checkpoint=infer_common_params['checkpoint'],
                  output_columns=output_columns,
                  output_file_name=infer_common_params['infer_filename'])

######################################
# Analyze Common Params
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS['infer_filename'] = INFER_COMMON_PARAMS['infer_filename']
######################################
# Analyze Template
######################################
def eval_template(paths: dict, eval_common_params: dict):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Eval', {'attrs': ['bold', 'underline']})
    

    # metrics
    metrics = {
        'roc': MetricROCCurve(pred='model.output.ClinSig', target='data.ground_truth',
                                  output_filename=os.path.join(paths['inference_dir'], 'roc_curve.png')),
        'auc': MetricAUCROC(pred='model.output.ClinSig', target='data.ground_truth')
    }

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(ids=None,
                            data=eval_common_params["infer_filename"],
                            metrics=metrics,
                            output_dir=paths["eval_dir"])

    return results
######################################
# Run
######################################
if __name__ == "__main__":

    # allocate gpus
    NUM_GPUS = 1
    if NUM_GPUS == 0:
        TRAIN_COMMON_PARAMS['manager.train_params']['device'] = 'cpu'
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    FuseUtilsGPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    RUNNING_MODES = ['train','infer', 'eval']  # Options: 'train', 'infer', 'eval'

    if 'train' in RUNNING_MODES:
        train_template(paths=PATHS, train_common_params=TRAIN_COMMON_PARAMS)

    if 'infer' in RUNNING_MODES:
        infer_template(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    if 'eval' in RUNNING_MODES:
        eval_template(paths=PATHS,eval_common_params=EVAL_COMMON_PARAMS)