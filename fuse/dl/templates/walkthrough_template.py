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
from typing import OrderedDict
import logging

import torch.optim as optim

from torch.utils.data.dataloader import DataLoader
from fuse.utils.utils_debug import FuseDebug
import fuse.utils.gpu as GPU
from fuse.utils.utils_logger import fuse_logger_start


from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault

from fuse.dl.models.model_default import ModelDefault
from fuse.dl.managers.callbacks.callback_tensorboard import TensorboardCallback
from fuse.dl.managers.callbacks.callback_metric_statistics import MetricStatisticsCallback
from fuse.dl.managers.callbacks.callback_time_statistics import TimeStatisticsCallback
from fuse.dl.managers.manager_default import ManagerDefault

from fuse.eval.evaluator import EvaluatorDefault

###########################################################################################################
# Fuse
# Example of a training template:
#
# Training template contains 6 fundamental building blocks:
# (1) Data - detailed README file can be found at [fuse/data](../../fuse/data)
# (2) Model
# (3) Losses
# (4) Metrics and Evaluation - detailed README file can be found at [fuse/eval](../../fuse/eval)
# (5) Callbacks
# (6) Manager
#
# The template will create each of those building blocks and will eventually run Manager.train()
#
# Terminology:
# 1. sample-id -
#    A unique identifier of a sample. Each sample in the dataset must have an id that uniquely identifies it.
#    Examples of sample ids:
#    * path to the image file
#    * Tuple of (provider_id, patient_id, image_id)
#    * Running index
# 2. sample_dict - 
#    Represents a single sample and contains all relevant information about the sample.
#    No specific structure of this dictionary is required, but a useful pattern is to split it into sections (keys that define a "namespace" ): such as "data", "model",  etc.
#    NDict (fuse/utils/ndict.py) class is used instead of python standard dictionary in order to allow easy "." separated access. For example:
#    `sample_dict[“data.input.img”]` is the equivalent of `sample_dict["data"]["input"]["img"]`
#    Another recommended convention is to include suffix specifying the type of the value ("img", "seg", "bbox")
# 3. epoch_result -
#    A dictionary created by the manager and includes the losses and metrics value
#    as defined within the template and calculated for the specific epoch.
# 4. Fuse base classes - *Base -
#    Abstract classes of the object forming together Fuse framework .
# 5. Fuse default classes - *Default -
#    A default generic implementation of the equivalent base class.
#    Those generic implementation will be useful for most common use cases.
#    Alternative implementations could be implemented for the special cases.
###########################################################################################################

##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'fast', 'debug', 'verbose', 'user'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################
# Output Paths
##########################################
model_dir = None # TODO: fill in a path to model dir
PATHS = {'model_dir': model_dir,
         'force_reset_model_dir': False,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'cache_dir': 'TODO',
         'inference_dir': os.path.join(model_dir, "infer"),
         'eval_dir': os.path.join(model_dir, "eval")}

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
TRAIN_COMMON_PARAMS['data.cache_num_workers'] = 10

# ===============
# Manager - Train
# ===============
TRAIN_COMMON_PARAMS['manager.train_params'] = {
    'num_epochs': 100,
    'virtual_batch_size': 1,  # number of batches in one virtual batch
    'start_saving_epochs': 10,  # first epoch to start saving checkpoints from
    'gap_between_saving_epochs': 5,  # number of epochs between saved checkpoint
    'lr_sch_target': "train.losses.total_loss" # key to a value in epoch results dictionary to pass to the learning rate scheduler. (typically: 'validation.losses.total_loss' or 'train.losses.total_loss')
}
TRAIN_COMMON_PARAMS['manager.best_epoch_source'] = {
    'source': 'TODO',  # can be any key from 'epoch_results' (either metrics or losses result)
    'optimization': 'max',  # can be either min/max
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
    #   
    #   Default dataset implementation built from a sequence of op(erator)s.
    #   Operators are the building blocks of the sample processing pipeline. 
    #   Each operator gets as input the *sample_dict* as created by the previous operators and can either add/delete/modify fields in sample_dict. 
    #   The operator interface is specified in OpBase class. 
    #   
    #   We split the pipeline into two parts - static and dynamic, which allow us to control the part out of the entire pipeline that will be cached. 
    # 
    #   For more details and examples, read (fuse/data/README.md)[../../fuse/data/README.md]
    #   A complete example implementation of a dataset can be bound in  (fuseimg/datasets/stoic21.py)[../../examples/fuse_examples/imaging/classification/stoic21/runner_stoic21.py] STOIC21.static_pipeline(). 
    # ==============================================================================

    #### Train Data

    lgr.info(f'Train Data:', {'attrs': 'bold'})

    ## TODO - list your sample ids:
    # Fuse TIP - splitting the sample_ids to folds can be done by fuse.data.utils.split.dataset_balanced_division_to_folds().
    #            See (examples/fuse_examples/imaging/classification/stoic21/runner_stoic21.py)[../../examples/fuse_examples/imaging/classification/stoic21/runner_stoic21.py] 
    train_sample_ids = None
    validation_sample_ids = None

    ## Create data static_pipeline - 
    #                                the output of this pipeline will be cached to optimize the running time and to better utilize the GPU:
    #                                See example in (fuseimg/datasets/stoic21.py)[../../examples/fuse_examples/imaging/classification/stoic21/runner_stoic21.py] STOIC21.static_pipeline().
    static_pipeline = PipelineDefault("template_static", [
        
    ])
    ## Create data dynamic_pipeline - Dynamic pipeline follows the static pipeline and continues to pre-process the sample. 
    #                                 In contrast to the static pipeline, the output of the dynamic pipeline is not be cached and allows modifying the pre-precessing steps without recaching, 
    #                                 The recommendation is to include in the dynamic pipeline pre-processing steps that we intend to experiment with or augmentation steps.
    train_dynamic_pipeline = PipelineDefault("template_dynamic", [

    ])
    validation_dynamic_pipeline = PipelineDefault("template_dynamic", [

    ])


    # Create dataset
    cacher = SamplesCacher(f'template_cache', 
            static_pipeline,
            [paths['cache_dir']], restart_cache=False, workers=train_common_params["data.cache_num_workers"])            

    train_dataset = DatasetDefault(sample_ids=train_sample_ids,
        static_pipeline=static_pipeline,
        dynamic_pipeline=train_dynamic_pipeline,
        cacher=cacher,            
    )

    lgr.info(f'- Load and cache data:')
    train_dataset.create()
    lgr.info(f'- Load and cache data: Done')

    ## Create batch sampler
    # Fuse TIPs:
    # 1. You don't have to balance according the classification labels, any categorical value will do.
    #    Use balanced_class_name to select the categorical value
    # 2. You don't have to equally balance between the classes.
    #    Use balanced_class_weights to specify the number of required samples in a batch per each class
    # 3. Use mode to specify probabilities rather then exact number of samples from  a class in each batch
    lgr.info(f'- Create sampler:')
    sampler = BatchSamplerDefault(dataset=train_dataset,
                                       balanced_class_name='TODO',
                                       num_balanced_classes='TODO',
                                       batch_size=train_common_params['data.batch_size'],
                                       balanced_class_weights=None)

    lgr.info(f'- Create sampler: Done')

    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=False, drop_last=False,
                                  batch_sampler=sampler,
                                  collate_fn=CollateDefault(),
                                  num_workers=train_common_params['data.train_num_workers'])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})

    #### Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})
    
    validation_dataset = DatasetDefault(sample_ids=validation_sample_ids,
        static_pipeline=static_pipeline,
        dynamic_pipeline=validation_dynamic_pipeline,
        cacher=cacher,            
    )
    

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
                                       collate_fn=CollateDefault())
    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})

    # ===================================================================================================================
    # Model
    #   Build a model (torch.nn.Module) using generic Fuse components:
    #   1. ModelDefault - generic component supporting single backbone with multiple heads
    #   2. Backbone - simple backbone model
    #   3. Head* - generic head implementations
    #   The model outputs will be aggregated in batch_dict['model.*']
    #   Each head output will be aggregated in batch_dict['model.<head name>.*']
    #
    #   Additional implemented models:
    #   * ModelWrapper - allow to use single standard PyTorch module as is - see (examples/fuse_examples/imaging/classification/mnist/runner.py)[../../examples/fuse_examples/imaging/classification/mnist/runner.py]
    #   * ModelEnsemble - runs several sub-modules sequentially
    #   * ModelMultistream - convolutional neural network with multiple processing streams and multiple heads
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
    #   LossDefault - wraps a PyTorch loss function with an api.
    #
    # ==========================================================================================================================================
    losses = {
        # TODO add losses here (instances of LossBase)
    }

    # =========================================================================================================
    # Metrics - details can be found in (fuse/eval/README.md)[../../fuse/eval/README.md]
    # =========================================================================================================
    metrics = OrderedDict([
        # TODO add metrics here (<name>, <instance of MetricBase>)

    ])
        
    # ==========================================================================================================
    #  Callbacks
    #  Callbacks are sub-classes of CallbackBase.
    #  A callback is an object that can perform actions at various stages of training,
    #  In each stage it allows to manipulate either the data, batch_dict or epoch_results.
    # ==========================================================================================================
    callbacks = [
        # Fuse TIPs: add additional callbacks here
        # default callbacks
        TensorboardCallback(model_dir=paths['model_dir']),  # save statistics for tensorboard
        MetricStatisticsCallback(output_path=paths['model_dir'] + "/metrics.csv"),  # save statistics in a csv file
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

    #### create infer dataset
    infer_dataset = None # TODO: follow the same steps to create dataset as in run_train
    
    ## Create dataloader
    infer_dataloader = DataLoader(dataset=infer_dataset,
                                       shuffle=False,
                                       drop_last=False,
                                       batch_sampler=None,
                                       batch_size=infer_common_params['data.batch_size'],
                                       num_workers=infer_common_params['data.num_workers'],
                                       collate_fn=CollateDefault())
    #### Manager for inference
    manager = ManagerDefault()
    # TODO - define the keys out of batch_dict that will be saved to a file
    output_columns = [None] # TODO: specify the key name out of the batch_dict to dump. Optionally also include the key of the target name
    manager.infer(data_loader=infer_dataloader,
                  input_model_dir=paths['model_dir'],
                  checkpoint=infer_common_params['checkpoint'],
                  output_columns=output_columns,
                  output_file_name=infer_common_params['infer_filename'])



######################################
# Eval Template
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS['infer_filename'] = INFER_COMMON_PARAMS['infer_filename']

def run_eval(paths: dict, eval_common_params: dict):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Eval', {'attrs': ['bold', 'underline']})

    # metrics
    metrics = OrderedDict([
        # TODO add metrics here (<name>, <instance of MetricBase>)
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
    NUM_GPUS = 1
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    RUNNING_MODES = ['train', 'infer', 'eval']  # Options: 'train', 'infer', 'eval'

    # train
    if 'train' in RUNNING_MODES:
        train_template(paths=PATHS, train_common_params=TRAIN_COMMON_PARAMS)

    # infer
    if 'infer' in RUNNING_MODES:
        infer_template(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

     # eval
    if 'eval' in RUNNING_MODES:
        run_eval(paths=PATHS, eval_common_params=EVAL_COMMON_PARAMS)
