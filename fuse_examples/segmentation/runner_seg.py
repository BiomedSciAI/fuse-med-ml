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
import logging
from glob import glob
import random
import numpy as np
import matplotlib.pylab as plt
from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from fuse.data.augmentor.augmentor_toolbox import aug_op_affine_group, aug_op_affine, aug_op_color, aug_op_gaussian, aug_op_elastic_transform
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerUniform as Uniform
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandBool as RandBool
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandInt as RandInt
from fuse.utils.utils_gpu import FuseUtilsGPU
from fuse.utils.utils_logger import fuse_logger_start
from fuse.data.augmentor.augmentor_default import FuseAugmentorDefault
from fuse.data.visualizer.visualizer_default import FuseVisualizerDefault
from fuse.data.dataset.dataset_default import FuseDatasetDefault
from fuse.models.model_wrapper import FuseModelWrapper
from fuse.losses.segmentation.loss_dice import DiceBCELoss
from fuse.losses.segmentation.loss_dice import FuseDiceLoss
from fuse.losses.loss_default import FuseLossDefault
from fuse.managers.manager_default import FuseManagerDefault
from fuse.managers.callbacks.callback_tensorboard import FuseTensorboardCallback
from fuse.managers.callbacks.callback_metric_statistics import FuseMetricStatisticsCallback
from fuse.managers.callbacks.callback_time_statistics import FuseTimeStatisticsCallback
from fuse.data.processor.processor_dataframe import FuseProcessorDataFrame
# from fuse.metrics.metric_auc_per_pixel import FuseMetricAUCPerPixel
# from fuse.metrics.segmentation.metric_score_map import FuseMetricScoreMap
# from fuse.analyzer.analyzer_default import FuseAnalyzerDefault
from fuse.eval.evaluator import EvaluatorDefault
from fuse.eval.metrics.segmentation.metrics_segmentation_common import MetricDice, MetricIouJaccard, MetricOverlap, Metric2DHausdorff, MetricPixelAccuracy

from data_source_segmentation import FuseDataSourceSeg
from seg_input_processor import SegInputProcessor

from unet import UNet


def perform_softmax(output):
    if isinstance(output, torch.Tensor):  # validation
        logits = output
    else:  # train
        logits = output.logits
    cls_preds = F.softmax(logits, dim=1)
    return logits, cls_preds


SZ = 512
TRAIN = f'../../data/siim/data{SZ}/train/'
TEST = f'../../data/siim/data{SZ}/test/'
MASKS = f'../../data/siim/data{SZ}/masks/'

# TODO: Path to save model
ROOT = '../results/'
# TODO: path to store the data  
ROOT_DATA = ROOT
# TODO: Name of the experiment
EXPERIMENT = 'unet_seg_results'
# TODO: Path to cache data
CACHE_PATH = '../results/'
# TODO: Name of the cached data folder
EXPERIMENT_CACHE = 'exp_cache'

PATHS = {'data_dir': [TRAIN, MASKS, TEST],
         'model_dir': os.path.join(ROOT, EXPERIMENT, 'model_dir'),
         'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'cache_dir': os.path.join(CACHE_PATH, EXPERIMENT_CACHE+'_cache_dir'),
         'inference_dir': os.path.join(ROOT, EXPERIMENT, 'infer_dir'),
         'analyze_dir': os.path.join(ROOT, EXPERIMENT, 'analyze_dir')}

##########################################
# Train Common Params
##########################################
# ============
# Data
# ============
TRAIN_COMMON_PARAMS = {}
TRAIN_COMMON_PARAMS['data.batch_size'] = 8
TRAIN_COMMON_PARAMS['data.train_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.augmentation_pipeline'] = [
    [
        ('data.input.input_0','data.gt.gt_global'),
        aug_op_affine_group,
        {'rotate': Uniform(-20.0, 20.0),  
        'flip': (RandBool(0.0), RandBool(0.5)),  # only flip right-to-left
        'scale': Uniform(0.9, 1.1),
        'translate': (RandInt(-50, 50), RandInt(-50, 50))},
        {'apply': RandBool(0.9)}
    ],
    [
        ('data.input.input_0','data.gt.gt_global'),
        aug_op_elastic_transform,
        {},
        {'apply': RandBool(0.7)}
    ],
    [
        ('data.input.input_0',),
        aug_op_color,
        {
         'add': Uniform(-0.06, 0.06), 
         'mul': Uniform(0.95, 1.05), 
         'gamma': Uniform(0.9, 1.1),
         'contrast': Uniform(0.85, 1.15)
        },
        {'apply': RandBool(0.7)}
    ],
    [
        ('data.input.input_0',),
        aug_op_gaussian,
        {'std': 0.05},
        {'apply': RandBool(0.7)}
    ],
]

# ===============
# Manager - Train1
# ===============
TRAIN_COMMON_PARAMS['manager.train_params'] = {
    'num_epochs': 2,
    'virtual_batch_size': 1,  # number of batches in one virtual batch
    'start_saving_epochs': 10,  # first epoch to start saving checkpoints from
    'gap_between_saving_epochs': 5,  # number of epochs between saved checkpoint
}
TRAIN_COMMON_PARAMS['manager.best_epoch_source'] = {
    'source': 'losses.total_loss',  # can be any key from 'epoch_results' (either metrics or losses result)
    'optimization': 'min',  # can be either min/max
}
TRAIN_COMMON_PARAMS['manager.learning_rate'] = 1e-1
TRAIN_COMMON_PARAMS['manager.weight_decay'] = 1e-4  
TRAIN_COMMON_PARAMS['manager.resume_checkpoint_filename'] = None  # if not None, will try to load the checkpoint
TRAIN_COMMON_PARAMS['partition_file'] = 'train_val_split.pickle'

#################################
# Train Template
#################################
def run_train(paths: dict, train_common_params: dict):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')

    # Download data
    # TODO - function to download + arrange the data

    lgr.info('\nFuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})

    train_path = paths['data_dir'][0]
    mask_path = paths['data_dir'][1]

    #### Train Data
    lgr.info(f'Train Data:', {'attrs': 'bold'})

    train_data_source = FuseDataSourceSeg(image_source=train_path,
                                          mask_source=mask_path,
                                          partition_file=train_common_params['partition_file'],
                                          train=True)
    print(train_data_source.summary())

    ## Create data processors:
    input_processors = {
        'input_0': SegInputProcessor(name='image')
    }
    gt_processors = {
        'gt_global': SegInputProcessor(name='mask')
    }

    ## Create data augmentation (optional)
    augmentor = FuseAugmentorDefault(augmentation_pipeline=train_common_params['data.augmentation_pipeline'])

    # Create visualizer (optional)
    visualiser = FuseVisualizerDefault(image_name='data.input.input_0', 
                                       mask_name='data.gt.gt_global',
                                       pred_name='model.logits.classification')

    train_dataset = FuseDatasetDefault(cache_dest=paths['cache_dir'],
                                       data_source=train_data_source,
                                       input_processors=input_processors,
                                       gt_processors=gt_processors,
                                       augmentor=augmentor,
                                       visualizer=visualiser)

    lgr.info(f'- Load and cache data:')
    train_dataset.create()
    lgr.info(f'- Load and cache data: Done')

    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=True, 
                                  drop_last=False,
                                  batch_size=train_common_params['data.batch_size'],
                                  collate_fn=train_dataset.collate_fn,
                                  num_workers=train_common_params['data.train_num_workers'])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})
    # ==================================================================
    # Validation dataset
    lgr.info(f'Validation Data:', {'attrs': 'bold'})

    valid_data_source = FuseDataSourceSeg(image_source=train_path,
                                          mask_source=mask_path,
                                          partition_file=train_common_params['partition_file'],
                                          train=False)
    print(valid_data_source.summary())

    valid_dataset = FuseDatasetDefault(cache_dest=paths['cache_dir'],
                                    data_source=valid_data_source,
                                    input_processors=input_processors,
                                    gt_processors=gt_processors,
                                    visualizer=visualiser)

    lgr.info(f'- Load and cache data:')
    valid_dataset.create()
    lgr.info(f'- Load and cache data: Done')

    ## Create dataloader
    validation_dataloader = DataLoader(dataset=valid_dataset,
                                       shuffle=False, 
                                       drop_last=False,
                                       batch_size=train_common_params['data.batch_size'],
                                       collate_fn=valid_dataset.collate_fn,
                                       num_workers=train_common_params['data.validation_num_workers'])

    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})
    # ==================================================================
    # # Training graph
    lgr.info('Model:', {'attrs': 'bold'})
    torch_model = UNet(n_channels=1, n_classes=1, bilinear=False)

    model = FuseModelWrapper(model=torch_model,
                            model_inputs=['data.input.input_0'],
                            post_forward_processing_function=perform_softmax,
                            model_outputs=['logits.classification', 'output.classification']
                            )

    lgr.info('Model: Done', {'attrs': 'bold'})
    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        'dice_loss': DiceBCELoss(pred_name='model.logits.classification', 
                                 target_name='data.gt.gt_global')
    }

    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), 
                          lr=train_common_params['manager.learning_rate'],
                          momentum=0.9,
                          weight_decay=train_common_params['manager.weight_decay'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # train from scratch
    manager = FuseManagerDefault(output_model_dir=paths['model_dir'], 
                                force_reset=paths['force_reset_model_dir'])

    # =====================================================================================
    #  Callbacks
    # =====================================================================================
    callbacks = [
        # default callbacks
        FuseTensorboardCallback(model_dir=paths['model_dir']),  # save statistics for tensorboard
        FuseMetricStatisticsCallback(output_path=paths['model_dir'] + "/metrics.csv"),  # save statistics in a csv file
        FuseTimeStatisticsCallback(num_epochs=train_common_params['manager.train_params']['num_epochs'], load_expected_part=0.1)  # time profiler
    ]

    # Providing the objects required for the training process.
    manager.set_objects(net=model,
                        optimizer=optimizer,
                        losses=losses,
                        lr_scheduler=scheduler,
                        callbacks=callbacks,
                        best_epoch_source=train_common_params['manager.best_epoch_source'],
                        train_params=train_common_params['manager.train_params'],
                        output_model_dir=paths['model_dir'])

    manager.train(train_dataloader=train_dataloader,
                  validation_dataloader=validation_dataloader)
    lgr.info('Train: Done', {'attrs': 'bold'})


######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS['infer_filename'] = os.path.join(PATHS['inference_dir'], 'validation_set_infer.gz')
INFER_COMMON_PARAMS['checkpoint'] = 'last'  # Fuse TIP: possible values are 'best', 'last' or epoch_index.
INFER_COMMON_PARAMS['data.train_num_workers'] = TRAIN_COMMON_PARAMS['data.train_num_workers']
INFER_COMMON_PARAMS['partition_file'] = TRAIN_COMMON_PARAMS['partition_file']
INFER_COMMON_PARAMS['data.batch_size'] = TRAIN_COMMON_PARAMS['data.batch_size']

######################################
# Inference Template
######################################
def run_infer(paths: dict, infer_common_params: dict):
    #### Logger
    fuse_logger_start(output_path=paths['inference_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={os.path.join(paths["inference_dir"], infer_common_params["infer_filename"])}', {'color': 'magenta'})

    train_path = paths['data_dir'][0]
    mask_path = paths['data_dir'][1]
    # ==================================================================
    # Validation dataset
    lgr.info(f'Test Data:', {'attrs': 'bold'})

    infer_data_source = FuseDataSourceSeg(image_source=train_path,
                                          mask_source=mask_path,
                                          partition_file=infer_common_params['partition_file'],
                                          train=False)
    print(infer_data_source.summary())

    ## Create data processors:
    input_processors = {
        'input_0': SegInputProcessor(name='image')
    }
    gt_processors = {
        'gt_global': SegInputProcessor(name='mask')
    }

    # Create visualizer (optional)
    visualiser = FuseVisualizerDefault(image_name='data.input.input_0', 
                                       mask_name='data.gt.gt_global',
                                       pred_name='model.logits.classification')

    infer_dataset = FuseDatasetDefault(cache_dest=paths['cache_dir'],
                                       data_source=infer_data_source,
                                       input_processors=input_processors,
                                       gt_processors=gt_processors,
                                       visualizer=visualiser)

    lgr.info(f'- Load and cache data:')
    infer_dataset.create()
    lgr.info(f'- Load and cache data: Done')

    ## Create dataloader
    infer_dataloader = DataLoader(dataset=infer_dataset,
                                  shuffle=False, 
                                  drop_last=False,
                                  batch_size=infer_common_params['data.batch_size'],
                                  collate_fn=infer_dataset.collate_fn,
                                  num_workers=infer_common_params['data.train_num_workers'])

    lgr.info(f'Test Data: Done', {'attrs': 'bold'})

    #### Manager for inference
    manager = FuseManagerDefault()
    # extract just the global classification per sample and save to a file
    output_columns = ['model.logits.classification', 'data.gt.gt_global']
    manager.infer(data_loader=infer_dataloader,
                  input_model_dir=paths['model_dir'],
                  checkpoint=infer_common_params['checkpoint'],
                  output_columns=output_columns,
                  output_file_name=infer_common_params["infer_filename"])

    # visualize the predictions
    infer_processor = FuseProcessorDataFrame(data_pickle_filename=infer_common_params['infer_filename'])
    descriptors_list = infer_processor.get_samples_descriptors()
    out_name = 'model.logits.classification'
    gt_name = 'data.gt.gt_global' 
    for desc in descriptors_list[:10]:
        data = infer_processor(desc)
        pred = np.squeeze(data[out_name])
        gt = np.squeeze(data[gt_name])
        _, ax = plt.subplots(1,2)
        ax[0].imshow(pred)
        ax[0].set_title('prediction')
        ax[1].imshow(gt)
        ax[1].set_title('gt')
        fn = os.path.join(paths["inference_dir"], Path(desc[0]).name)
        plt.savefig(fn)

######################################
# Analyze Common Params
######################################
ANALYZE_COMMON_PARAMS = {}
ANALYZE_COMMON_PARAMS['infer_filename'] = INFER_COMMON_PARAMS['infer_filename']
ANALYZE_COMMON_PARAMS['output_filename'] = os.path.join(PATHS['analyze_dir'], 'all_metrics.txt')
ANALYZE_COMMON_PARAMS['num_workers'] = 4
ANALYZE_COMMON_PARAMS['batch_size'] = 8

######################################
# Analyze Template
######################################
def run_analyze(paths: dict, analyze_common_params: dict):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Analyze', {'attrs': ['bold', 'underline']})

    # gt_processors = {
    #     'gt_global': SegInputProcessor(name='mask')
    # }

    # metrics
    # {
    #     'auc': FuseMetricAUCPerPixel(pred_name='model.logits.classification', 
    #                                  target_name='data.gt.gt_global'), 
    #     'seg': FuseMetricScoreMap(pred_name='model.logits.classification', 
    #                               target_name='data.gt.gt_global',
    #                               hard_threshold=True, threshold=0.5)
    # }

    metrics = OrderedDict([
            ("dice", MetricDice(pred='model.logits.classification', 
                                target='data.gt.gt_global')),
    ])

    # create analyzer
    evaluator = EvaluatorDefault()

    results = evaluator.eval(ids=None, 
                             data=analyze_common_params['infer_filename'],
                             metrics=metrics) 

#     # run
#     analyzer.analyze(gt_processors=gt_processors,
#                      data_pickle_filename=analyze_common_params['infer_filename'],
#                      metrics=metrics,
#                      print_results=True,
#                      output_filename=analyze_common_params['output_filename'],
#                      num_workers=0) 


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

    RUNNING_MODES = ['train', 'infer', 'analyze']  # Options: 'train', 'infer', 'analyze'
    RUNNING_MODES = ['analyze']  # Options: 'train', 'infer', 'analyze'

    # train
    if 'train' in RUNNING_MODES:
        run_train(paths=PATHS, train_common_params=TRAIN_COMMON_PARAMS)

    # infer
    if 'infer' in RUNNING_MODES:
        run_infer(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    # analyze
    if 'analyze' in RUNNING_MODES:
        run_analyze(paths=PATHS, analyze_common_params=ANALYZE_COMMON_PARAMS)

