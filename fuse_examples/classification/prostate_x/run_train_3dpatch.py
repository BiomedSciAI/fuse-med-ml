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
from functools import partial
from multiprocessing import Manager
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from fuse.metrics.classification.metric_auc import FuseMetricAUC
from fuse.data.augmentor.augmentor_default import FuseAugmentorDefault
from fuse.data.augmentor.augmentor_toolbox import unsqueeze_2d_to_3d, aug_op_color, aug_op_affine, squeeze_3d_to_2d, \
    rotation_in_3d
from fuse.data.dataset.dataset_generator import FuseDatasetGenerator
from fuse.data.sampler.sampler_balanced_batch import FuseSamplerBalancedBatch
from fuse.losses.loss_default import FuseLossDefault
from fuse.managers.callbacks.callback_metric_statistics import FuseMetricStatisticsCallback
from fuse.managers.callbacks.callback_tensorboard import FuseTensorboardCallback
from fuse.managers.callbacks.callback_time_statistics import FuseTimeStatisticsCallback
from fuse.managers.manager_default import FuseManagerDefault

from fuse.utils.utils_gpu import FuseUtilsGPU
from fuse.utils.utils_gpu import FuseUtilsGPU as gpu
from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandBool as RandBool
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandInt as RandInt
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerUniform as Uniform
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerChoice


from fuse_examples.classification.prostate_x.backbone_3d_multichannel import Fuse_model_3d_multichannel,ResNet
from fuse_examples.classification.prostate_x.patient_data_source import FuseProstateXDataSourcePatient
from fuse_examples.classification.prostate_x.processor import  FuseProstateXPatchProcessor
from fuse_examples.classification.prostate_x.tasks import FuseProstateXTask
from fuse_examples.classification.prostate_x.head_1d_classifier import FuseHead1dClassifier
from fuse_examples.classification.prostate_x.post_processor import post_processing



##########################################
# Output Paths
# ##########################################

# TODO: path to save model
root_path = ''
# TODO: path for prostateX data
root_data = ''

PATHS = {'force_reset_model_dir': False,
         # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'model_dir': root_path + '3D/fold0_T2_DWI_ADC_Ktrans_2/',
         'cache_dir': root_path + 'cache_fold0_T2_DWI_ADC_Ktrans/',
         'dataset_dir':root_path,

         'prostate_data_path' : root_data,
         'ktrans_path': root_data + 'ProstateXKtrains-train-fixed/',
         }

#################################
# Train Template
#################################


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
    #### Train Data

    lgr.info(f'Train Data:', {'attrs': 'bold'})

    ## Create data source:
    DATABASE_REVISION = 29062021
    lgr.info(f'database_revision={DATABASE_REVISION}', {'color': 'magenta'})

    # create data source
    train_data_source = FuseProstateXDataSourcePatient(PATHS['dataset_dir'],'train',
                                                  db_ver=DATABASE_REVISION, db_name = train_common_params['db_name'],
                                                  fold_no = train_common_params['fold_no'], include_gt=False)

    ## Create data processors:
    image_processing_args = {
        'patch_xy': 74,
        'patch_z': 13,
    }

    ## Create data processor

    generate_processor = FuseProstateXPatchProcessor(path_to_db=PATHS['dataset_dir'],
                                                data_path=PATHS['prostate_data_path'],
                                                ktrans_data_path=PATHS['ktrans_path'],
                                                db_name=train_common_params['db_name'],
                                                fold_no=train_common_params['fold_no'],
                                                lsn_shape=(image_processing_args['patch_z'],
                                                           image_processing_args['patch_xy'],
                                                           image_processing_args['patch_xy']),
                                                                 )

    train_post_processor = partial(post_processing)

    # data augmentation (optional)
    num_channels =TRAIN_COMMON_PARAMS['backbone_model_dict']['input_channels_num']
    slice_num = image_processing_args['patch_z']

    image_channels = [list(range(0, slice_num))]
    mask_channels = [[i] for i in range(slice_num, slice_num * 2)]
    aug_pipeline = [
        [
            ('data.input',),
            rotation_in_3d,
            {'z_rot': Uniform(-5.0, 5.0), 'y_rot': Uniform(-5.0, 5.0), 'x_rot': Uniform(-5.0, 5.0)},
            {'apply': RandBool(0.5)}
        ],
        [
            ('data.input',),
            squeeze_3d_to_2d,
            {'axis_squeeze': 'z'},
            {}
        ],
        [
            ('data.input',),
            aug_op_affine,
            {'rotate': Uniform(0, 360.0),
             'translate':(RandInt(-4, 4), RandInt(-4, 4)),
             'flip': (RandBool(0.5), RandBool(0.5)),
             'scale': Uniform(0.9, 1.1),
             },
            {'apply': RandBool(0.5)}
        ],
        [
            ('data.input',),
            aug_op_affine,
            {'rotate': Uniform(-3.0, 3.0),
             'translate': (RandInt(-2, 2), RandInt(-2, 2)),
             'flip': (False, False),
             'scale': Uniform(0.9, 1.1),
             'channels': FuseUtilsParamSamplerChoice(image_channels, probabilities=None)},
            {'apply': RandBool(0.5) if train_common_params['data.aug.phase_misalignment'] else 0}
        ],

        [
            ('data.input',),
            aug_op_color,
            {'add': Uniform(-0.06, 0.06),
             'mul': Uniform(0.95, 1.05),
             'gamma': Uniform(0.9, 1.1),
             'contrast': Uniform(0.85, 1.15)},
            {'apply': RandBool(0.5)}
        ],

        [
            ('data.input',),
            unsqueeze_2d_to_3d,
            {'channels': num_channels, 'axis_squeeze': 'z'},
            {}
        ],
    ]
    augmentor = FuseAugmentorDefault(augmentation_pipeline=aug_pipeline)

    # Create dataset
    train_dataset = FuseDatasetGenerator(cache_dest=paths['cache_dir'],
                                         data_source=train_data_source,
                                         processor=generate_processor,
                                         post_processing_func=train_post_processor,
                                         augmentor=augmentor,
                                         statistic_keys=['data.ground_truth']
                                         )

    gpu_ids_for_caching = []
    lgr.info(f'- Load and cache data:')
    with Manager() as mp_manager:
        # Use gpu or cpu to generate the data - one per process - use gpus to speed it up
        gpu_list = mp_manager.list(gpu_ids_for_caching)
        train_dataset.create(num_workers=len(gpu_list), worker_init_func=FuseUtilsGPU.allocate_gpu_for_process,
                             worker_init_args=(gpu_list,))

    lgr.info(f'- Load and cache data: Done')

    #### Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})

    ## Create data source
    validation_data_source = FuseProstateXDataSourcePatient(PATHS['dataset_dir'],'validation',
                                                       db_ver=DATABASE_REVISION ,
                                                       db_name = train_common_params['db_name'],
                                                       fold_no=train_common_params['fold_no'], include_gt=False)

    # post processor
    validation_post_processor = partial(post_processing)

    ## Create dataset
    validation_dataset = FuseDatasetGenerator(cache_dest=paths['cache_dir'],
                                              data_source=validation_data_source,
                                              processor=generate_processor,
                                              post_processing_func=validation_post_processor,
                                              augmentor=None,
                                              statistic_keys=['data.ground_truth']
                                              )

    lgr.info(f'- Load and cache data:')
    with Manager() as mp_manager:
        # Use gpu or cpu to generate the data - one per process - use gpus to speed it up
        gpu_list = mp_manager.list(gpu_ids_for_caching)
        validation_dataset.create(num_workers=len(gpu_list), worker_init_func=FuseUtilsGPU.allocate_gpu_for_process,
                                  worker_init_args=(gpu_list,))#len(gpu_list)
    lgr.info(f'Data - task caching and filtering:', {'attrs': 'bold'})


    ## Create dataloader
    lgr.info(f'- Create sampler:')

    sampler = FuseSamplerBalancedBatch(dataset=train_dataset,
                                       balanced_class_name='data.ground_truth',
                                       num_balanced_classes=train_common_params['multi_phase_task_loss'].num_classes(),
                                       batch_size=train_common_params['data.batch_size'],
                                       balanced_class_weights=[25] * train_common_params[
                                           'multi_phase_task'].num_classes(),
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
        backbone= ResNet(),
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

        'auc': FuseMetricAUC(pred_name='model.output.ClinSig', target_name='data.ground_truth',
                                class_names=train_common_params['multi_phase_task'].class_names()),
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

    ##########################################
    # Train Common Params
    ##########################################
    # ============
    # Data
    # ============
    TRAIN_COMMON_PARAMS = {}
    TRAIN_COMMON_PARAMS['db_name']='prostate_x'
    TRAIN_COMMON_PARAMS['fold_no']=0
    TRAIN_COMMON_PARAMS['data.batch_size'] = 50
    TRAIN_COMMON_PARAMS['data.train_num_workers'] = 10
    TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 10
    # add misalignment to segmentation
    TRAIN_COMMON_PARAMS['data.aug.mask_misalignment'] = True
    # add misalignment to phase registration
    TRAIN_COMMON_PARAMS['data.aug.phase_misalignment'] = True

    # ===============
    # Manager - Train
    # ===============
    TRAIN_COMMON_PARAMS['manager.train_params'] = {
        'num_gpus': 1,
        'num_epochs': 150,
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
    TRAIN_COMMON_PARAMS['multi_phase_task_loss'] = FuseProstateXTask('ClinSig', 0)
    TRAIN_COMMON_PARAMS['multi_phase_task'] = FuseProstateXTask('ClinSig', 0)


    ##

    TRAIN_COMMON_PARAMS['mode'] = 'single_chan'  #
    # additional backbone that extract global features from a full slice which then concatenated to volume patch features
    TRAIN_COMMON_PARAMS['append_global_slice'] = False
    # global features to concat after the global max pooling
    TRAIN_COMMON_PARAMS[
        'append_features_to_gmp'] = None#[('data.tensor_clinical', 3)
    TRAIN_COMMON_PARAMS['append_features_to_image'] =  (None,0)

    TRAIN_COMMON_PARAMS['ch_inx_to_use'] = []
    if TRAIN_COMMON_PARAMS['append_global_slice']:
        TRAIN_COMMON_PARAMS['append_features_to_gmp'].append(
            ('model.backbone_global_features', TRAIN_COMMON_PARAMS['num_backbone_features']))
    # backbone parameters
    TRAIN_COMMON_PARAMS['backbone_model_dict'] = \
        {'input_channels_num': 5,
         }
    # train
    train_template(paths=PATHS, train_common_params=TRAIN_COMMON_PARAMS)
