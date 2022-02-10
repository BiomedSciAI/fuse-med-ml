
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

import os

from fuse.utils.utils_debug import FuseUtilsDebug
from fuse.utils.utils_gpu import FuseUtilsGPU

import logging

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from fuse.utils.utils_logger import fuse_logger_start

from fuse.data.sampler.sampler_balanced_batch import FuseSamplerBalancedBatch

from fuse.models.model_default import FuseModelDefault
from fuse.models.heads.head_global_pooling_classifier import FuseHeadGlobalPoolingClassifier

from fuse.losses.loss_default import FuseLossDefault

from fuse.metrics.classification.metric_auc import FuseMetricAUC
from fuse.metrics.classification.metric_accuracy import FuseMetricAccuracy

from fuse.managers.callbacks.callback_tensorboard import FuseTensorboardCallback
from fuse.managers.callbacks.callback_metric_statistics import FuseMetricStatisticsCallback
from fuse.managers.callbacks.callback_time_statistics import FuseTimeStatisticsCallback
from fuse.managers.manager_default import FuseManagerDefault

from fuse_examples.classification.MG_CMMD.dataset import CMMD_2021_dataset
from fuse.models.backbones.backbone_inception_resnet_v2 import FuseBackboneInceptionResnetV2


from fuse.analyzer.analyzer_default import FuseAnalyzerDefault
from fuse.metrics.classification.metric_roc_curve import FuseMetricROCCurve
##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'fast', 'debug', 'verbose', 'user'. See details in FuseUtilsDebug
debug = FuseUtilsDebug(mode)


##########################################
# Train Common Params
##########################################
# ============
# Data
# ============
TRAIN_COMMON_PARAMS = {}

TRAIN_COMMON_PARAMS['data.train_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 8

######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS['infer_filename'] = 'validation_set_infer.gz'
INFER_COMMON_PARAMS['checkpoint'] = 'best'  # Fuse TIP: possible values are 'best', 'last' or epoch_index.
INFER_COMMON_PARAMS['data.train_num_workers'] = TRAIN_COMMON_PARAMS['data.train_num_workers']
# ===============
# Manager - Train
# ===============
NUM_GPUS = 2
TRAIN_COMMON_PARAMS['data.batch_size'] = 2 *NUM_GPUS
TRAIN_COMMON_PARAMS['manager.train_params'] = {
    'num_gpus': NUM_GPUS,
    'num_epochs': 100,
    'virtual_batch_size': 1,  # number of batches in one virtual batch
    'start_saving_epochs': 10,  # first epoch to start saving checkpoints from
    'gap_between_saving_epochs': 100,  # number of epochs between saved checkpoint
}

# best_epoch_source
# if an epoch values are the best so far, the epoch is saved as a checkpoint.
TRAIN_COMMON_PARAMS['manager.best_epoch_source'] = {
    'source': 'metrics.auc.macro_avg',  # can be any key from losses or metrics dictionaries
    'optimization': 'max',  # can be either min/max
    'on_equal_values': 'better',
    # can be either better/worse - whether to consider best epoch when values are equal
}
TRAIN_COMMON_PARAMS['manager.learning_rate'] = 1e-5
TRAIN_COMMON_PARAMS['manager.weight_decay'] = 0.001
TRAIN_COMMON_PARAMS['manager.resume_checkpoint_filename'] = None
#################################
# Train Template
#################################
def run_train(paths: dict, train_common_params: dict, reset_cache: bool):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')

    # Download data
    # TBD

    lgr.info('\nFuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})

    #### Train Data
    lgr.info(f'Train Data:', {'attrs': 'bold'})
    train_dataset, validation_dataset , _ = CMMD_2021_dataset(paths['data_dir'], paths['data_misc_dir'], reset_cache=reset_cache)

    ## Create sampler
    lgr.info(f'- Create sampler:')
    sampler = FuseSamplerBalancedBatch(dataset=train_dataset,
                                       balanced_class_name='data.gt.classification',
                                       num_balanced_classes=2,
                                       batch_size=train_common_params['data.batch_size'])

    lgr.info(f'- Create sampler: Done')

    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=False, drop_last=False,
                                  batch_sampler=sampler, collate_fn=train_dataset.collate_fn,
                                  num_workers=train_common_params['data.train_num_workers'])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})

    #### Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})




    ## Create dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       shuffle=False,
                                       drop_last=False,
                                       batch_sampler=None,
                                       batch_size=train_common_params['data.batch_size'],
                                       num_workers=train_common_params['data.validation_num_workers'],
                                       collate_fn=validation_dataset.collate_fn)
    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})

    # ===================================================================
    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info('Model:', {'attrs': 'bold'})

    model = FuseModelDefault(
        conv_inputs=(('data.input.image', 1),),
        backbone=FuseBackboneInceptionResnetV2(input_channels_num=1),
        heads=[
            FuseHeadGlobalPoolingClassifier(head_name='head_0',
                                            dropout_rate=0.5,
                                            conv_inputs=[('model.backbone_features', 384)],
                                            layers_description=(256,),
                                            num_classes=2,
                                            pooling="avg"),
        ]
    )

    lgr.info('Model: Done', {'attrs': 'bold'})


    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        'cls_loss': FuseLossDefault(pred_name='model.logits.head_0', target_name='data.gt.classification',
                                    callable=F.cross_entropy, weight=1.0)
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    metrics = {
        'auc': FuseMetricAUC(pred_name='model.output.head_0', target_name='data.gt.classification'),
        'accuracy': FuseMetricAccuracy(pred_name='model.output.head_0', target_name='data.gt.classification')
    }

    # =====================================================================================
    #  Callbacks
    # =====================================================================================
    callbacks = [
        # default callbacks
        FuseTensorboardCallback(model_dir=paths['model_dir']),  # save statistics for tensorboard
        FuseMetricStatisticsCallback(output_path=paths['model_dir'] + "/metrics.csv"),  # save statistics in a csv file
        FuseTimeStatisticsCallback(num_epochs=train_common_params['manager.train_params']['num_epochs'], load_expected_part=0.1)  # time profiler
    ]

    # =====================================================================================
    #  Manager - Train
    #  Create a manager, training objects and run a training process.
    # =====================================================================================
    lgr.info('Train:', {'attrs': 'bold'})

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=train_common_params['manager.learning_rate'],
                           weight_decay=train_common_params['manager.weight_decay'])

    # create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

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
                        train_params=train_common_params['manager.train_params'],
                        output_model_dir=paths['model_dir'])

    # Continue training
    if train_common_params['manager.resume_checkpoint_filename'] is not None:
        # Loading the checkpoint including model weights, learning rate, and epoch_index.
        manager.load_checkpoint(checkpoint=train_common_params['manager.resume_checkpoint_filename'], mode='train',
                                values_to_resume=['net'])
    # # Start training
    manager.train(train_dataloader=train_dataloader,
                  validation_dataloader=validation_dataloader)

    lgr.info('Train: Done', {'attrs': 'bold'})

######################################
# Inference Template
######################################
def run_infer(paths: dict, infer_common_params: dict):
    #### Logger
    fuse_logger_start(output_path=paths['inference_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={os.path.join(paths["inference_dir"], infer_common_params["infer_filename"])}', {'color': 'magenta'})

    # Create data source:
    _, _ , test_dataset = CMMD_2021_dataset(paths['data_dir'])


    ## Create dataloader
    infer_dataloader = DataLoader(dataset=test_dataset,
                                  shuffle=False, drop_last=False,
                                  collate_fn=test_dataset.collate_fn,
                                  num_workers=infer_common_params['data.train_num_workers'])
    lgr.info(f'Test Data: Done', {'attrs': 'bold'})

    #### Manager for inference
    manager = FuseManagerDefault()
    # extract just the global classification per sample and save to a file
    output_columns = ['model.output.head_0','data.gt.classification']
    manager.infer(data_loader=infer_dataloader,
                  input_model_dir=paths['model_dir'],
                  checkpoint=infer_common_params['checkpoint'],
                  output_columns=output_columns,
                  output_file_name=os.path.join(paths["inference_dir"], infer_common_params["infer_filename"]))
    
    ######################################
# Analyze Common Params
######################################
ANALYZE_COMMON_PARAMS = {}
ANALYZE_COMMON_PARAMS['infer_filename'] = INFER_COMMON_PARAMS['infer_filename']
ANALYZE_COMMON_PARAMS['output_filename'] = 'all_metrics.txt'
ANALYZE_COMMON_PARAMS['num_workers'] = 4
ANALYZE_COMMON_PARAMS['batch_size'] = 8


######################################
# Analyze Template
######################################
def run_analyze(paths: dict, analyze_common_params: dict):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Analyze', {'attrs': ['bold', 'underline']})

     # metrics
    metrics = {
        'accuracy': FuseMetricAccuracy(pred_name='model.output.classification', target_name='data.gt.classification'),
        'roc': FuseMetricROCCurve(pred_name='model.output.classification', target_name='data.gt.classification', output_filename=os.path.join(paths['inference_dir'], 'roc_curve.png')),
        'auc': FuseMetricAUC(pred_name='model.output.classification', target_name='data.gt.classification')
    }

    # create analyzer
    analyzer = FuseAnalyzerDefault()

    # run
    # FIXME: simplify analyze interface for this case
    results = analyzer.analyze(gt_processors={},
                     data_pickle_filename=os.path.join(paths["inference_dir"], analyze_common_params["infer_filename"]),
                     metrics=metrics,
                     output_filename=analyze_common_params['output_filename'])

    return results
######################################
# Run
######################################


if __name__ == "__main__":
    # allocate gpus
    if NUM_GPUS == 0:
        TRAIN_COMMON_PARAMS['manager.train_params']['device'] = 'cpu'
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    FuseUtilsGPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    RUNNING_MODES = ['train', 'infer', 'analyze']  # Options: 'train', 'infer', 'analyze'
    # Path to save model
    root = ''
    # Path to the stored CMMD dataset location
    # dataset should be download from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508
    # download requires NBIA data retriever https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
    # put on the folliwing in the main folder  - 
    # 1. CMMD_clinicaldata_revision.csv which is a converted version of CMMD_clinicaldata_revision.xlsx 
    # 2. folder named CMMD which is the downloaded data folder
    root_data = None #TODO: add path to the data folder
    assert root_data is not None, "Error: please set root_data, the path to the stored CMMD dataset location"
    # Name of the experiment
    experiment = 'model_new/CMMD_classification'
    # Path to cache data
    cache_path = 'examples/'
    # Name of the cached data folder
    experiment_cache = 'CMMD_'
    paths = {'data_dir': root_data,
             'model_dir': os.path.join(root, experiment, 'model_dir_transfer'),
             'force_reset_model_dir': True,
             # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
             'cache_dir': os.path.join(cache_path, experiment_cache + '_cache_dir'),
             'inference_dir': os.path.join(root, experiment, 'infer_dir')}
    # train
    if 'train' in RUNNING_MODES:
        run_train(paths=paths, train_common_params=TRAIN_COMMON_PARAMS, reset_cache=False)

    # infer
    if 'infer' in RUNNING_MODES:
        run_infer(paths=paths, infer_common_params=INFER_COMMON_PARAMS)
    #
    # analyze
    if 'analyze' in RUNNING_MODES:
        run_analyze(paths=paths, analyze_common_params=ANALYZE_COMMON_PARAMS)
