
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
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds

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

from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC, MetricAccuracy, MetricROCCurve

from fuse.managers.callbacks.callback_tensorboard import FuseTensorboardCallback
from fuse.managers.callbacks.callback_metric_statistics import FuseMetricStatisticsCallback
from fuse.managers.callbacks.callback_time_statistics import FuseTimeStatisticsCallback
from fuse.managers.manager_default import FuseManagerDefault

from fuse_examples.classification.cmmd.dataset import CMMD_2021_dataset
from fuse.models.backbones.backbone_inception_resnet_v2 import FuseBackboneInceptionResnetV2

import hydra
from omegaconf import DictConfig, OmegaConf
from fuse.eval.evaluator import EvaluatorDefault
##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'fast', 'debug', 'verbose', 'user'. See details in FuseUtilsDebug
debug = FuseUtilsDebug(mode)


#################################
# Train Template
#################################
def run_train(cfg : DictConfig):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=cfg.paths.model_dir, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')

    # Download data
    # TBD

    lgr.info('\nFuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={cfg.paths.model_dir}', {'color': 'magenta'})
    lgr.info(f'cache_dir={cfg.paths.cache_dir}', {'color': 'magenta'})

    #### Train Data
    lgr.info(f'Train Data:', {'attrs': 'bold'})
    train_dataset, validation_dataset , _ = CMMD_2021_dataset(cfg.paths.data_dir, cfg.paths.data_misc_dir, reset_cache=cfg.params.reset_cache)

    ## Create sampler
    lgr.info(f'- Create sampler:')
    sampler = FuseSamplerBalancedBatch(dataset=train_dataset,
                                       balanced_class_name='data.gt.classification',
                                       num_balanced_classes=2,
                                       batch_size=cfg.train.batch_size)

    lgr.info(f'- Create sampler: Done')

    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=False, drop_last=False,
                                  batch_sampler=sampler, collate_fn=train_dataset.collate_fn,
                                  num_workers=cfg.train.num_workers)
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})

    #### Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})




    ## Create dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       shuffle=False,
                                       drop_last=False,
                                       batch_sampler=None,
                                       batch_size=cfg.train.batch_size,
                                       num_workers=cfg.train.num_workers,
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
    metrics = OrderedDict([
        ('op', MetricApplyThresholds(pred='model.output.head_0')), # will apply argmax
        ('auc', MetricAUCROC(pred='model.output.head_0', target='data.gt.classification')),
        ('accuracy', MetricAccuracy(pred='results:metrics.op.cls_pred', target='data.gt.classification')),
    ])

    # =====================================================================================
    #  Callbacks
    # =====================================================================================
    callbacks = [
        # default callbacks
        FuseTensorboardCallback(model_dir=cfg.paths.model_dir),  # save statistics for tensorboard
        FuseMetricStatisticsCallback(output_path=cfg.paths.model_dir + "/metrics.csv"),  # save statistics in a csv file
        FuseTimeStatisticsCallback(num_epochs=cfg.train.manager_train_params.num_epochs, load_expected_part=0.1)  # time profiler
    ]

    # =====================================================================================
    #  Manager - Train
    #  Create a manager, training objects and run a training process.
    # =====================================================================================
    lgr.info('Train:', {'attrs': 'bold'})

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate,
                           weight_decay=cfg.train.weight_decay)

    # create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # train from scratch
    manager = FuseManagerDefault(output_model_dir=cfg.paths.model_dir, force_reset=cfg.params.force_reset_model_dir)
    # Providing the objects required for the training process.
    manager.set_objects(net=model,
                        optimizer=optimizer,
                        losses=losses,
                        metrics=metrics,
                        best_epoch_source=OmegaConf.to_container(cfg.train.manager_best_epoch_source),
                        lr_scheduler=scheduler,
                        callbacks=callbacks,
                        train_params=OmegaConf.to_container(cfg.train.manager_train_params),
                        output_model_dir=cfg.paths.model_dir)

    # Continue training
    if cfg.train.resume_checkpoint_filename is not None:
        # Loading the checkpoint including model weights, learning rate, and epoch_index.
        manager.load_checkpoint(checkpoint=cfg.train.resume_checkpoint_filename, mode='train',
                                values_to_resume=['net'])
    # # Start training
    manager.train(train_dataloader=train_dataloader,
                  validation_dataloader=validation_dataloader)

    lgr.info('Train: Done', {'attrs': 'bold'})

######################################
# Inference Template
######################################
def run_infer(cfg : DictConfig):
    #### Logger
    fuse_logger_start(output_path=cfg.paths.inference_dir, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={os.path.join(cfg.paths.inference_dir, cfg.infer.infer_filename)}', {'color': 'magenta'})

    # Create data source:
    _, _ , test_dataset = CMMD_2021_dataset(cfg.paths.data_dir, cfg.paths.data_misc_dir)


    ## Create dataloader
    infer_dataloader = DataLoader(dataset=test_dataset,
                                  shuffle=False, drop_last=False,
                                  collate_fn=test_dataset.collate_fn,
                                  num_workers=cfg.infer.num_workers)
    lgr.info(f'Test Data: Done', {'attrs': 'bold'})

    #### Manager for inference
    manager = FuseManagerDefault()
    # extract just the global classification per sample and save to a file
    output_columns = ['model.output.head_0','data.gt.classification']
    manager.infer(data_loader=infer_dataloader,
                  input_model_dir=cfg.paths.model_dir,
                  checkpoint=cfg.infer.checkpoint,
                  output_columns=output_columns,
                  output_file_name=os.path.join(cfg.paths.inference_dir, cfg.infer.infer_filename))
    


######################################
# Analyze Template
######################################
def run_eval(cfg : DictConfig):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Eval', {'attrs': ['bold', 'underline']})

     # metrics
    metrics = OrderedDict([
        ('op', MetricApplyThresholds(pred='model.output.head_0')), # will apply argmax
        ('auc', MetricAUCROC(pred='model.output.head_0', target='data.gt.classification')),
        ('accuracy', MetricAccuracy(pred='results:metrics.op.cls_pred', target='data.gt.classification')),
    ])

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(ids=None,
                     data=os.path.join(cfg.paths.inference_dir, cfg.infer.infer_filename),
                     metrics=metrics,
                     output_dir=cfg.paths.eval_dir)

    return results

@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print(cfg)
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    FuseUtilsGPU.choose_and_enable_multiple_gpus(cfg.train.manager_train_params.num_gpus, force_gpus=force_gpus)

    RUNNING_MODES = ['train', 'infer', 'analyze']  # Options: 'train', 'infer', 'analyze'
    # Path to the stored dataset location
    # dataset should be download from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508
    # download requires NBIA data retriever https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
    # put on the folliwing in the main folder  - 
    # 1. CMMD_clinicaldata_revision.csv which is a converted version of CMMD_clinicaldata_revision.xlsx 
    # 2. folder named CMMD which is the downloaded data folder

    # train
    if 'train' in RUNNING_MODES:
        run_train(cfg)

    # infer
    if 'infer' in RUNNING_MODES:
        run_infer(cfg)
    #
    # analyze
    if 'analyze' in RUNNING_MODES:
        run_eval(cfg)
if __name__ == "__main__":
    main()
