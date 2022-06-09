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

from fuse.utils.utils_debug import FuseDebug
import fuse.utils.gpu as GPU

os.environ['skip_broker'] = '1'

import logging

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from fuse.utils.utils_logger import fuse_logger_start

from fuse.data.sampler.sampler_balanced_batch import SamplerBalancedBatch
from fuse.data.visualizer.visualizer_default import VisualizerDefault
from fuse.data.augmentor.augmentor_default import AugmentorDefault
from fuse.data.dataset.dataset_default import DatasetDefault

from fuse.dl.models.model_default import ModelDefault

from fuse.dl.managers.callbacks.callback_tensorboard import TensorboardCallback
from fuse.dl.managers.callbacks.callback_metric_statistics import StatisticsCallback
from fuse.dl.managers.callbacks.callback_time_statistics import TimeStatisticsCallback
from fuse.dl.managers.manager_default import ManagerDefault

from fuse.eval.evaluator import EvaluatorDefault

###########################################################################################################
# Fuse
# Example of a training template:
#
# Training template contains 6 fundamental building blocks:
# (1) Data
# (2) Model
# (3) Losses
# (4) Metrics
# (5) Callbacks
# (6) Manager
#
# The template will create each of those building and will eventually run Manager.train()
#
# Terminology:
# 1.batch_dict -
#   Dictionary that aggregates data and results during the batch cycle.
#   The resulted batch is hierarchical dictionary including 3 branches: 'data', 'model', 'losses'
#   batch_dict allows us to decouple the training components.
#   For example 'classification_loss' code don't directly interact with the data or model,
#   Instead a string name, which is a key in batch dict, will be specified for both the:
#   (1) classification label: 'data.gt.global.label' (2) model logits predictions: 'model.head_0.logits'
# 2.epoch_result -
#   A dictionary created by the manager and includes the losses and metrics value
#   as defined within the template and calculated for the specific epoch
# 3.Fuse base classes - Fuse*Base -
#   Abstract classes of the object forming together Fuse framework .
# 4.Fuse default classes - Fuse*Default -
#   A default generic implementation of the equivalent base class.
#   Those generic implementation will be useful for most common use cases.
#   Alternative implementations could be implemented for the special cases.
# 5.sample_descriptor -
#   Unique ID representing sample
###########################################################################################################

##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################
# Output Paths
##########################################
PATHS = {'model_dir': 'TODO',
         'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'cache_dir': 'TODO',
         'inference_dir': 'TODO'}

##########################################
# Train Common Params
##########################################
# ============
# Data
# ============
TRAIN_COMMON_PARAMS = {}
TRAIN_COMMON_PARAMS['data.batch_size'] = 2
TRAIN_COMMON_PARAMS['data.train_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.augmentation_pipeline'] = [
    # TODO: define the augmentation pipeline here
    # Fuse TIP: Use as a reference the simple augmentation pipeline written in Fuse.data.augmentor.augmentor_toolbox.aug_image_default_pipeline
]
# ===============
# Manager - Train
# ===============
TRAIN_COMMON_PARAMS['manager.train_params'] = {
    'num_epochs': 100,
    'virtual_batch_size': 1,  # number of batches in one virtual batch
    'start_saving_epochs': 10,  # first epoch to start saving checkpoints from
    'gap_between_saving_epochs': 5,  # number of epochs between saved checkpoint
}
TRAIN_COMMON_PARAMS['manager.best_epoch_source'] = {
    'source': 'TODO',  # can be any key from 'epoch_results' (either metrics or losses result)
    'optimization': 'max',  # can be either min/max
    'on_equal_values': 'better',
    # can be either better/worse - whether to consider best epoch when values are equal
}
TRAIN_COMMON_PARAMS['manager.learning_rate'] = 1e-4
TRAIN_COMMON_PARAMS['manager.weight_decay'] = 0.001
TRAIN_COMMON_PARAMS['manager.resume_checkpoint_filename'] = None  # if not None, will try to load the checkpoint


#################################
# Train Template
#################################
def train_template(paths: dict, train_common_params: dict):
    # ==============================================================================
    # Logger
    #   - output log automatically to three destinations:
    #     (1) console (2) file - copy of the console (3) verbose file - used for debug
    #   - save a copy of the template file
    # ==============================================================================
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})

    # ==============================================================================
    # Data
    #   Build dataloaders (torch.utils.data.DataLoader) for both train and validation.
    #   Using Fuse generic components:
    #
    #   (1) Data Source - DataSourceBase -
    #       Class providing list of sample descriptors
    #
    #   (2) Processor - ProcessorBase -
    #       Group of classes extracting sample data given the sample descriptor.
    #       We divide the processor to two groups (1) 'input' (2) 'gt'.
    #       Both groups will be used for training, but only the 'input' group will be used for 'inference'
    #       The input processors data will be aggregated in batch_dict['data.input.<processor name>.*']
    #       The gt processors data will be aggregated in batch_dict['data.gt.<processor name>.*']
    #       Available Processor classes are:
    #       ProcessorCSV -  Reads CSV file, call() returns a sample as dict
    #       ProcessorDataFrame -  Reads either data frame or pickled data frame, call() returns a sample as dict
    #       ProcessorDataFrameWithGT - Reads either data frame or pickled data frame, call() returns a gt tensor value as dict
    #       ProcessorRand - Processor generating random ground truth - useful for testing and sanity check
    #
    #   (3) Dataset - DatasetBase -
    #       Extended pytorch Dataset class - providing additional functionality such as caching, filtering, visualizing.
    #       Available Dataset classes are:
    #       DatasetDefault - generic default implementation
    #       DatasetGenerator - to be used when generating simple samples at once (e.g., patches of a single image)
    #       DatasetWrapper - wraps Pytorch's dataset, converts each sample into a dict.
    #
    #   (4) Augmentor - AugmentorBase -
    #       Optional class applying the augmentation
    #       See AugmentorDefault for default generic implementation of augmentor. It is aimed to be used by most experiments.
    #       See fuse.data.augmentor.augmentor_toolbox.py for implemented augmentation functions to be used in the pipeline.
    #
    #   (5) Visualizer - VisualizerBase -
    #       Optional class visualizing the data before and after augmentations
    #       Available visualizers:
    #       VisualizerDefault - Visualizer for data including single 2D image with optional mask
    #       Fuse3DVisualizerDefault - Visualizer for data including 3D volume with optional mask
    #       VisualizerImageAnalysis - Visualizer for producing analysis of an image
    #
    #   (6) Sampler - implementing 'torch.utils.data.sampler' -
    #       Class retrieving list of samples to use for each batch
    #       Available Sampler;
    #       SamplerBalancedBatch - balances data per batch. Supports balancing of classes by weights/probabilities.
    # ==============================================================================
    #### Train Data

    lgr.info(f'Train Data:', {'attrs': 'bold'})

    ## Create data source:
    # TODO: Create instance of DataSourceBase
    # Reference: MGDataSource
    train_data_source = None

    ## Create data processors:
    # TODO - Create instances of ProcessorBase and add to the dictionaries below
    # Reference: ProcessorDataFrame, ProcessorCSV, DatasetProcessor (for pytorch data)
    input_processors = {}
    gt_processors = {}

    ## Create data augmentation (optional)
    augmentor = AugmentorDefault(augmentation_pipeline=train_common_params['data.augmentation_pipeline'])

    # Create visualizer (optional)
    # TODO - Either use the default visualizer or an alternative one
    visualiser = VisualizerDefault(image_name='TODO', label_name='TODO')

    # Create dataset
    # Fuse TIP: If it's more convenient to generate few samples at once, use DatasetGenerator
    train_dataset = DatasetDefault(cache_dest=paths['cache_dir'],
                                       data_source=train_data_source,
                                       input_processors=input_processors,
                                       gt_processors=gt_processors,
                                       augmentor=augmentor,
                                       visualizer=visualiser)

    lgr.info(f'- Load and cache data:')
    train_dataset.create()
    lgr.info(f'- Load and cache data: Done')

    ## Fuse TIPs:
    # 1. Get to know the resulted data structure train_dataset[0]
    # 2. Visualize here the input to the network using 'train_dataset.visualize(sample_index)
    # 3. Compare augmented case to non-augmented case using 'train_dataset.visualize_augmentation(sample_index)'
    # 4. Review the summary generated by train_dataset.summary() including basic statistics

    ## Create sampler
    # Fuse TIPs:
    # 1. You don't have to balance according the classification labels, any categorical value will do.
    #    Use balanced_class_name to select the categorical value
    # 2. You don't have to equally balance between the classes.
    #    Use balanced_class_weights to specify the number of required samples in a batch per each class
    lgr.info(f'- Create sampler:')
    sampler = SamplerBalancedBatch(dataset=train_dataset,
                                       balanced_class_name='TODO',
                                       num_balanced_classes='TODO',
                                       batch_size=train_common_params['data.batch_size'],
                                       balanced_class_weights=None)

    lgr.info(f'- Create sampler: Done')

    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=False, drop_last=False,
                                  batch_sampler=sampler,
                                  collate_fn=train_dataset.collate_fn,
                                  num_workers=train_common_params['data.train_num_workers'])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})

    #### Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})

    ## Create data source
    # TODO: Create instance of DataSourceBase
    # Reference: DataSourceDefault
    validation_data_source = 'TODO'

    ## Create dataset
    # Fuse TIP: If it's more convenient to generate few samples at once, use DatasetGenerator
    validation_dataset = DatasetDefault(cache_dest=paths['cache_dir'],
                                            data_source=validation_data_source,
                                            input_processors=input_processors,
                                            gt_processors=gt_processors,
                                            augmentor=None,
                                            visualizer=visualiser)

    lgr.info(f'- Load and cache data:')
    validation_dataset.create()
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

    # ===================================================================================================================
    # Model
    #   Build a model (torch.nn.Module) using generic Fuse componnets:
    #   1. ModelDefault - generic component supporting single backbone with multiple heads
    #   2. Backbone - simple backbone model
    #   3. Head* - generic head implementations
    #   The model outputs will be aggregated in batch_dict['model.*']
    #   Each head output will be aggregated in batch_dict['model.<head name>.*']
    #
    #   Additional implemented models:
    #   * ModelEnsemble - runs several sub-modules sequentially
    #   * ModelMultistream - convolutional neural network with multiple processing streams and multiple heads
    #   *
    # ===================================================================================================================
    lgr.info('Model:', {'attrs': 'bold'})
    # TODO - define / create a model
    model = ModelDefault(
        conv_inputs=(('data.input.input_0.tensor', 1),),
        backbone='TODO',  # Reference: BackboneInceptionResnetV2
        heads=['TODO']  # References: HeadGlobalPoolingClassifier, HeadDenseSegmentation
    )
    lgr.info('Model: Done', {'attrs': 'bold'})

    # ==========================================================================================================================================
    #   Loss
    #   Dictionary of loss elements each element is a sub-class of LossBase
    #   The total loss will be the weighted sum of all the elements.
    #   Each element output loss will be aggregated in batch_dict['losses.<loss name>']
    #   The average batch loss per epoch will be included in epoch_result['losses.<loss name>'],
    #   and the total loss in epoch_result['losses.total_loss']
    #   The 'best_epoch_source', used to save the best model could be based on one of this losses.
    #   Available Losses:
    #   LossDefault - wraps a PyTorch loss function with a Fuse api.
    #   LossSegmentationCrossEntropy - calculates cross entropy loss per location ("dense") of a class activation map ("segmentation")
    #
    # ==========================================================================================================================================
    losses = {
        # TODO add losses here (instances of LossBase)
    }

    # =========================================================================================================
    # Metrics
    # Dictionary of metric elements. Each element is a sub-class of MetricBase
    # The metrics will be calculated per epoch for both the validation and train.
    # The results will be included  in epoch_result['metrics.<metric name>']
    # The 'best_epoch_source', used to save the best model could be based on one of this metrics.
    # Available Metrics:
    # See fuse/eva;/README.md for more details
    #
    # =========================================================================================================
    metrics = {
        # TODO add metrics here (instances of MetricBase)
    }

    # ==========================================================================================================
    #  Callbacks
    #  Callbacks are sub-classes of CallbackBase.
    #  A callback is an object that can perform actions at various stages of training,
    #  In each stage it allows to manipulate either the data, batch_dict or epoch_results.
    # ==========================================================================================================
    callbacks = [
        # Fuse TIPs: add additional callbacks here
        # default callbacks
        TensorboardCallback(model_dir=paths['model_dir']),  # save statstics for tensorboard
        MetricStatisticsCallback(output_path=paths['model_dir'] + "/metrics.csv"),  # save statisticsin a csv file
        TimeStatisticsCallback(num_epochs=train_common_params['manager.train_params']['num_epochs'], load_expected_part=0.1)  # time profiler
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
    manager = ManagerDefault(output_model_dir=paths['model_dir'], force_reset=paths['force_reset_model_dir'])
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
INFER_COMMON_PARAMS['infer_filename'] = os.path.join(PATHS['inference_dir'], 'validation_set_infer.pickle')
INFER_COMMON_PARAMS['checkpoint'] = 'best'  # Fuse TIP: possible values are 'best', 'last' or epoch_index.


######################################
# Inference Template
######################################
def infer_template(paths: dict, infer_common_params: dict):
    #### Logger
    fuse_logger_start(output_path=paths['inference_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={infer_common_params["infer_filename"]}', {'color': 'magenta'})

    #### create infer datasource
    # TODO: Create instance of DataSourceBase
    # Reference: DataSourceDefault
    infer_data_source = 'TODO'
    lgr.info(f'experiment={infer_common_params["experiment_filename"]}', {'color': 'magenta'})

    #### Manager for inference
    manager = ManagerDefault()
    # TODO - define the keys out of batch_dict that will be saved to a file
    output_columns = ['TODO']
    manager.infer(data_source=infer_data_source,
                  input_model_dir=paths['model_dir'],
                  checkpoint=infer_common_params['checkpoint'],
                  output_columns=output_columns,
                  output_file_name=infer_common_params['infer_filename'])




######################################
# Run
######################################
if __name__ == "__main__":
    # allocate gpus
    NUM_GPUS = 1
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    RUNNING_MODES = ['train']  # Options: 'train', 'infer', 'analyze'

    # train
    if 'train' in RUNNING_MODES:
        train_template(paths=PATHS, train_common_params=TRAIN_COMMON_PARAMS)

    # infer
    if 'infer' in RUNNING_MODES:
        infer_template(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

 