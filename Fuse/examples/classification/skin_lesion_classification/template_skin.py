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

from Fuse.analyzer.analyzer_default import FuseAnalyzerDefault
from Fuse.utils.utils_debug import FuseUtilsDebug

os.environ['skip_broker'] = '1'

import logging

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from Fuse.utils.utils_param_sampler import FuseUtilsParamSamplerUniform as Uniform
from Fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandInt as RandInt
from Fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandBool as RandBool
from Fuse.utils.utils_logger import fuse_logger_start

from Fuse.data.sampler.sampler_balanced_batch import FuseSamplerBalancedBatch
from Fuse.data.visualizer.visualizer_default import FuseVisualizerDefault
from Fuse.data.augmentor.augmentor_default import FuseAugmentorDefault
from Fuse.data.augmentor.augmentor_toolbox import aug_op_affine, aug_op_color, aug_op_gaussian
from Fuse.data.dataset.dataset_default import FuseDatasetDefault

from Fuse.models.model_default import FuseModelDefault
from Fuse.models.backbones.Resnet import FuseBackboneResnet
from Fuse.models.heads.head_global_pooling_classifier import FuseHeadGlobalPoolingClassifier

from Fuse.losses.loss_default import FuseLossDefault

from Fuse.metrics.classification.metric_auc import FuseMetricAUC
from Fuse.metrics.classification.metric_accuracy import FuseMetricAccuracy
from Fuse.metrics.classification.metric_roc_curve import FuseMetricROCCurve

from Fuse.managers.callbacks.callback_tensorboard import FuseTensorboardCallback
from Fuse.managers.callbacks.callback_metric_statistics import FuseMetricStatisticsCallback
from Fuse.managers.callbacks.callback_time_statistics import FuseTimeStatisticsCallback
from Fuse.managers.manager_default import FuseManagerDefault


from Fuse.examples.classification.skin_lesion_classification.fuse_skin_data_source import FuseSkinDataSource
from Fuse.examples.classification.skin_lesion_classification import FuseSkinInputProcessor
from Fuse.examples.classification.skin_lesion_classification import FuseSkinGroundTruthProcessor
from Fuse.examples.classification.skin_lesion_classification.Inception import InceptionResnetV2
from Fuse.examples.classification.skin_lesion_classification.download import download_and_extract_isic


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
TRAIN_COMMON_PARAMS['data.year'] = '2017'  # year of the ISIC dataset

TRAIN_COMMON_PARAMS['data.train_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 8

TRAIN_COMMON_PARAMS['data.augmentation_pipeline'] = [
    [
        ('data.input.input_0',),
        aug_op_affine,
        {'rotate': Uniform(-180.0, 180.0), 'translate': (RandInt(-50, 50), RandInt(-50, 50)),
         'flip': (RandBool(0.3), RandBool(0.3)), 'scale': Uniform(0.9, 1.1)},
        {'apply': RandBool(0.9)}
    ],
    [
        ('data.input.input_0',),
        aug_op_color,
        {'add': Uniform(-0.06, 0.06), 'mul': Uniform(0.95, 1.05), 'gamma': Uniform(0.9, 1.1),
         'contrast': Uniform(0.85, 1.15)},
        {'apply': RandBool(0.7)}
    ],
    [
        ('data.input.input_0',),
        aug_op_gaussian,
        {'std': 0.03},
        {'apply': RandBool(0.7)}
    ],
]

assert TRAIN_COMMON_PARAMS['data.year'] == '2016' or TRAIN_COMMON_PARAMS['data.year'] == '2017'


##########################################
# Output Paths
##########################################

# TODO: Path to save model
ROOT = ''
# TODO: path to store the data
ROOT_DATA = ROOT
# TODO: Name of the experiment
EXPERIMENT = 'InceptionResnetV2_2017'
# TODO: Path to cache data
CACHE_PATH = ''
# TODO: Name of the cached data folder
EXPERIMENT_CACHE = 'ISIC_'+TRAIN_COMMON_PARAMS['data.year']

PATHS = {'data_dir': ROOT_DATA,
         'model_dir': os.path.join(ROOT, EXPERIMENT, 'model_dir'),
         'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'cache_dir': os.path.join(CACHE_PATH, EXPERIMENT_CACHE+'_cache_dir'),
         'inference_dir': os.path.join(ROOT, EXPERIMENT, 'infer_dir'),
         'analyze_dir': os.path.join(ROOT, EXPERIMENT, 'analyze_dir')}

# ===============
# Manager - Train
# ===============
num_gpus = 1
TRAIN_COMMON_PARAMS['data.batch_size'] = 8 * num_gpus
TRAIN_COMMON_PARAMS['manager.train_params'] = {
    'num_gpus': num_gpus,
    'num_epochs': 15,
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


#################################
# Train Template
#################################
def train_template(paths: dict, train_common_params: dict):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')

    # Download data
    download_and_extract_isic(paths['data_dir'], train_common_params['data.year'])

    lgr.info('\nFuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})

    #### Train Data

    lgr.info(f'Train Data:', {'attrs': 'bold'})

    ## Create data source:

    input_source_gt = {'2016': os.path.join(paths['data_dir'], 'data/ISIC2016_Training_GroundTruth.csv'),
                       '2017': os.path.join(paths['data_dir'], 'data/ISIC2017_Training_GroundTruth.csv')}
    training_data = {'2016': os.path.join(paths['data_dir'], 'data/ISIC2016_Training_Data/'),
                     '2017': os.path.join(paths['data_dir'], 'data/ISIC2017_Training_Data/')}
    input_source_gt_val = {'2016': input_source_gt['2016'],
                           '2017': os.path.join(paths['data_dir'], 'data/ISIC2017_Validation_GroundTruth.csv')}
    validation_data = {'2016': training_data['2016'],
                       '2017': os.path.join(paths['data_dir'], 'data/ISIC2017_Validation_Data/')}

    partition_file = {'2016': "train_val_split.pickle",
                      '2017': None}  # as no validation set is available in 2016 dataset, it needs to be created

    train_data_source = FuseSkinDataSource(input_source_gt[train_common_params['data.year']],
                                           partition_file=partition_file[train_common_params['data.year']],
                                           train=True,
                                           override_partition=True)

    ## Create data processors:
    input_processors = {
        'input_0': FuseSkinInputProcessor(input_data=training_data[train_common_params['data.year']])
    }
    gt_processors = {
        'gt_global': FuseSkinGroundTruthProcessor(input_data=input_source_gt[train_common_params['data.year']],
                                                  year=train_common_params['data.year'])
    }

    # Create data augmentation (optional)
    augmentor = FuseAugmentorDefault(
        augmentation_pipeline=train_common_params['data.augmentation_pipeline'])

    # Create visualizer (optional)
    visualiser = FuseVisualizerDefault(image_name='data.input.input_0', label_name='data.gt.gt_global.tensor')

    # Create dataset
    train_dataset = FuseDatasetDefault(cache_dest=paths['cache_dir'],
                                       data_source=train_data_source,
                                       input_processors=input_processors,
                                       gt_processors=gt_processors,
                                       augmentor=augmentor,
                                       visualizer=visualiser)

    lgr.info(f'- Load and cache data:')
    train_dataset.create()
    lgr.info(f'- Load and cache data: Done')

    ## Create sampler
    lgr.info(f'- Create sampler:')
    sampler = FuseSamplerBalancedBatch(dataset=train_dataset,
                                       balanced_class_name='data.gt.gt_global.tensor',
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

    ## Create data source
    validation_data_source = FuseSkinDataSource(input_source_gt_val[train_common_params['data.year']],
                                                partition_file=partition_file[train_common_params['data.year']],
                                                train=False)

    input_processors = {
        'input_0': FuseSkinInputProcessor(input_data=validation_data[train_common_params['data.year']])
    }
    gt_processors = {
        'gt_global': FuseSkinGroundTruthProcessor(input_data=input_source_gt_val[train_common_params['data.year']],
                                                  year=train_common_params['data.year'])
    }

    ## Create dataset
    validation_dataset = FuseDatasetDefault(cache_dest=paths['cache_dir'],
                                            data_source=validation_data_source,
                                            input_processors=input_processors,
                                            gt_processors=gt_processors,
                                            augmentor=None,
                                            visualizer=visualiser)

    lgr.info(f'- Load and cache data:')
    validation_dataset.create(pool_type='thread')  # use ThreadPool to create this dataset, to avoid cv2 problems in multithreading
    lgr.info(f'- Load and cache data: Done')

    ## Create dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       shuffle=False,
                                       drop_last=False,
                                       batch_sampler=None,
                                       batch_size=train_common_params['data.batch_size'],
                                       num_workers=train_common_params['data.validation_num_workers'],
                                       collate_fn=validation_dataset.collate_fn)
    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})

    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info('Model:', {'attrs': 'bold'})

    model = FuseModelDefault(
        conv_inputs=(('data.input.input_0', 1),),
        backbone={'Resnet18': FuseBackboneResnet(pretrained=True, in_channels=3, name='resnet18'),
                  'InceptionResnetV2': InceptionResnetV2()}['InceptionResnetV2'],
        heads=[
            FuseHeadGlobalPoolingClassifier(head_name='head_0',
                                            dropout_rate=0.5,
                                            conv_inputs=[('model.backbone_features', 1536)],
                                            post_concat_inputs=None,
                                            num_classes=2,
                                            pooling="avg"),
        ]
    )
    lgr.info('Model: Done', {'attrs': 'bold'})


    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        'cls_loss': FuseLossDefault(pred_name='model.logits.head_0', target_name='data.gt.gt_global.tensor',
                                    callable=F.cross_entropy, weight=1.0)
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    metrics = {
        'auc': FuseMetricAUC(pred_name='model.output.head_0', target_name='data.gt.gt_global.tensor'),
        'accuracy': FuseMetricAccuracy(pred_name='model.output.head_0', target_name='data.gt.gt_global.tensor')
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
    scheduler = {'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                 'CosineAnnealing': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)}['ReduceLROnPlateau']

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

    # Start training
    manager.train(train_dataloader=train_dataloader,
                  validation_dataloader=validation_dataloader)

    lgr.info('Train: Done', {'attrs': 'bold'})



######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS['infer_filename'] = os.path.join(PATHS['inference_dir'], 'validation_set_infer.gz')
INFER_COMMON_PARAMS['checkpoint'] = 'best'  # Fuse TIP: possible values are 'best', 'last' or epoch_index.
INFER_COMMON_PARAMS['data.year'] = TRAIN_COMMON_PARAMS['data.year']
INFER_COMMON_PARAMS['data.train_num_workers'] = TRAIN_COMMON_PARAMS['data.train_num_workers']


######################################
# Inference Template
######################################
def infer_template(paths: dict, infer_common_params: dict):
    #### Logger
    fuse_logger_start(output_path=paths['inference_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={infer_common_params["infer_filename"]}', {'color': 'magenta'})

    # Create data source:
    input_source = {'2016': os.path.join(paths['data_dir'], 'data/ISIC2016_Test_GroundTruth.csv'),
                    '2017': os.path.join(paths['data_dir'], 'data/ISIC2017_Test_GroundTruth.csv')}
    infer_data = {'2016': os.path.join(paths['data_dir'], 'data/ISIC2016_Test_Data/'),
                  '2017': os.path.join(paths['data_dir'], 'data/ISIC2017_Test_Data/')}

    infer_data_source = FuseSkinDataSource(input_source[infer_common_params['data.year']])

    # Create data processors
    input_processors_infer = {
        'input_0': FuseSkinInputProcessor(input_data=infer_data[infer_common_params['data.year']])
    }
    gt_processors_infer = {
        'gt_global': FuseSkinGroundTruthProcessor(input_data=input_source[infer_common_params['data.year']],
                                                  train=False, year=infer_common_params['data.year'])
    }

    # Create visualizer (optional)
    visualiser = FuseVisualizerDefault(image_name='data.input.input_0', label_name='data.gt.gt_global.tensor')

    # Create dataset
    infer_dataset = FuseDatasetDefault(cache_dest=paths['cache_dir'],
                                       data_source=infer_data_source,
                                       input_processors=input_processors_infer,
                                       gt_processors=gt_processors_infer,
                                       visualizer=visualiser)
    infer_dataset.create()

    lgr.info(f'- Create sampler: Done')

    ## Create dataloader
    infer_dataloader = DataLoader(dataset=infer_dataset,
                                  shuffle=False, drop_last=False,
                                  collate_fn=infer_dataset.collate_fn,
                                  num_workers=infer_common_params['data.train_num_workers'])
    lgr.info(f'Test Data: Done', {'attrs': 'bold'})

    #### Manager for inference
    manager = FuseManagerDefault()
    # extract just the global classification per sample and save to a file
    output_columns = ['model.output.head_0']
    manager.infer(data_loader=infer_dataloader,
                  input_model_dir=paths['model_dir'],
                  checkpoint=infer_common_params['checkpoint'],
                  output_columns=output_columns,
                  output_file_name=infer_common_params['infer_filename'])


######################################
# Analyze Common Params
######################################
ANALYZE_COMMON_PARAMS = {}
ANALYZE_COMMON_PARAMS['infer_filename'] = INFER_COMMON_PARAMS['infer_filename']
ANALYZE_COMMON_PARAMS['output_filename'] = os.path.join(PATHS['analyze_dir'], 'all_metrics.txt')
ANALYZE_COMMON_PARAMS['num_workers'] = 4
ANALYZE_COMMON_PARAMS['batch_size'] = 8
ANALYZE_COMMON_PARAMS['data.year'] = TRAIN_COMMON_PARAMS['data.year']


######################################
# Analyze Template
######################################
def analyze_template(paths: dict, analyze_common_params: dict):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Analyze', {'attrs': ['bold', 'underline']})

    # dataset generating ground truth processors
    input_source = {'2016': os.path.join(paths['data_dir'], 'data/ISIC2016_Test_GroundTruth.csv'),
                    '2017': os.path.join(paths['data_dir'], 'data/ISIC2017_Test_GroundTruth.csv')}

    ## Create data processors
    gt_processors = {
        'gt_global': FuseSkinGroundTruthProcessor(input_data=input_source[analyze_common_params['data.year']],
                                                  train=False, year=analyze_common_params['data.year'])
    }

    # metrics
    metrics = {
        'auc': FuseMetricAUC(pred_name='model.output.head_0', target_name='data.gt.gt_global.tensor'),
        'accuracy': FuseMetricAccuracy(pred_name='model.output.head_0', target_name='data.gt.gt_global.tensor'),
        'roc': FuseMetricROCCurve(pred_name='model.output.head_0', target_name='data.gt.gt_global.tensor',
                                  output_filename='roc_curve')
    }

    # create analyzer
    analyzer = FuseAnalyzerDefault()

    # run
    analyzer.analyze(gt_processors=gt_processors,
                     data_pickle_filename=analyze_common_params['infer_filename'],
                     metrics=metrics,
                     output_filename=analyze_common_params['output_filename'],
                     num_workers=analyze_common_params['num_workers'],
                     batch_size=analyze_common_params['num_workers'])


######################################
# Run
######################################


if __name__ == "__main__":

    RUNNING_MODES = ['train', 'infer', 'analyze']  # Options: 'train', 'infer', 'analyze'

    # train
    if 'train' in RUNNING_MODES:
        train_template(paths=PATHS, train_common_params=TRAIN_COMMON_PARAMS)

    # infer
    if 'infer' in RUNNING_MODES:
        infer_template(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    # analyze
    if 'analyze' in RUNNING_MODES:
        analyze_template(paths=PATHS, analyze_common_params=ANALYZE_COMMON_PARAMS)