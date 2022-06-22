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
import copy
import logging
import os
from typing import Any, Dict, OrderedDict, Sequence
from unittest import result
from fuse.dl.lightning.pl import LightningModuleDefault, convert_predictions_to_dataframe, model_checkpoint_callbacks
from fuse.dl.losses.loss_base import LossBase
from fuse.eval.metrics.metrics_common import MetricBase
from fuse.utils.file_io.file_io import create_dir, save_dataframe

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint 
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from fuse.eval.evaluator import EvaluatorDefault 
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve

from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault

from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict

from fuse.utils.utils_debug import FuseDebug
import fuse.utils.gpu as GPU
from fuse.utils.utils_logger import fuse_logger_start

from fuseimg.datasets.mnist import MNIST

from fuse_examples.imaging.classification.mnist import lenet
###########################################################################################################
# Fuse
###########################################################################################################

##########################################
# Lightning Module
##########################################
class LightningModuleMnist(LightningModuleDefault):
    def __init__(self, model_dir: str, opt_lr: float, opt_weight_decay: float, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model_dir"])

        self._model_dir = model_dir
        self._opt_lr = opt_lr
        self._opt_weight_decay = opt_weight_decay

        self.configure()

    def configure_models(self):
        torch_model = lenet.LeNet()

        model = ModelWrapSeqToDict(model=torch_model,
                            model_inputs=["data.image"],
                            post_forward_processing_function=perform_softmax,
                            model_outputs=['logits.classification', 'output.classification']
                            )
        return model
    
    def configure_losses(self) -> Dict[str, LossBase]:
        losses = {
            'cls_loss': LossDefault(pred='model.logits.classification', target='data.label', callable=F.cross_entropy, weight=1.0),
        }
        return losses
    
    def configure_train_metrics(self) -> OrderedDict[str, MetricBase]:
        metrics = OrderedDict([
            ('operation_point', MetricApplyThresholds(pred='model.output.classification')), # will apply argmax
            ('accuracy', MetricAccuracy(pred='results:metrics.operation_point.cls_pred', target='data.label'))
        ])
        return metrics
    
    def configure_validation_metrics(self) -> OrderedDict[str, MetricBase]:
        metrics = OrderedDict([
            ('operation_point', MetricApplyThresholds(pred='model.output.classification')), # will apply argmax
            ('accuracy', MetricAccuracy(pred='results:metrics.operation_point.cls_pred', target='data.label'))
        ])
        return metrics
    
    def configure_callbacks(self) -> Sequence[pl.Callback]:
        best_epoch_source = dict(
            monitor="validation.metrics.accuracy",
            mode="max",
        )
        return model_checkpoint_callbacks(self._model_dir, best_epoch_source)
    
    def configure_optimizers(self) -> Any:
        # create optimizer
        optimizer = optim.Adam(self._model.parameters(), lr=self._opt_lr, weight_decay=self._opt_weight_decay)

        # create learning scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        lr_sch_config = dict(scheduler=lr_scheduler,
                            monitor="validation.losses.total_loss")
        return dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################
# Output Paths
##########################################
ROOT = 'examples' # TODO: fill path here
PATHS = {'model_dir': os.path.join(ROOT, 'mnist/model_dir'),
         'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'cache_dir': os.path.join(ROOT, 'mnist/cache_dir'),
         'eval_dir': os.path.join(ROOT, 'mnist/eval_dir')}

##########################################
# Train Common Params
##########################################
TRAIN_COMMON_PARAMS = {}

# ============
# Data
# ============
TRAIN_COMMON_PARAMS['data.batch_size'] = 100
TRAIN_COMMON_PARAMS['data.train_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 8

# ===============
# Trainer
# ===============
TRAIN_COMMON_PARAMS['trainer.num_epochs'] = 2
TRAIN_COMMON_PARAMS['trainer.num_devices'] = 1
TRAIN_COMMON_PARAMS['trainer.accelerator'] = "gpu"

# ===============
# Checkpoint
# ===============
TRAIN_COMMON_PARAMS['checkpoint.resume_checkpoint_filename'] = None  # if not None, will try to load the checkpoint
# ===============
# Optimizer
# ===============
TRAIN_COMMON_PARAMS['opt.learning_rate'] = 1e-4
TRAIN_COMMON_PARAMS['opt.weight_decay'] = 0.001


def perform_softmax(output):
    if isinstance(output, torch.Tensor):  # validation
        logits = output
    else:  # train
        logits = output.logits
    cls_preds = F.softmax(logits, dim=1)
    return logits, cls_preds

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

    train_dataset = MNIST.dataset(paths["cache_dir"], train=True)
    lgr.info(f'- Create sampler:')
    sampler = BatchSamplerDefault(dataset=train_dataset,
                                       balanced_class_name='data.label',
                                       num_balanced_classes=10,
                                       batch_size=train_params['data.batch_size'],
                                       balanced_class_weights=None)
    lgr.info(f'- Create sampler: Done')

    # Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=sampler, collate_fn=CollateDefault(), num_workers=train_params['data.train_num_workers'])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})

    ## Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})
    # wrapping torch dataset
    validation_dataset = MNIST.dataset(paths["cache_dir"], train=False)
    
    # dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=train_params['data.batch_size'], collate_fn=CollateDefault(),
                                       num_workers=train_params['data.validation_num_workers'])
    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})



    lgr.info('Train:', {'attrs': 'bold'})


    pl_module = LightningModuleMnist(model_dir=paths["model_dir"], 
                                     opt_lr=train_params["opt.learning_rate"], 
                                     opt_weight_decay=train_params["opt.learning_rate"])
                

    pl_trainer = pl.Trainer(default_root_dir=paths['model_dir'],
                            max_epochs=train_params['trainer.num_epochs'],
                            accelerator=train_params["trainer.accelerator"],
                            devices=train_params["trainer.num_devices"],
                            auto_select_gpus=True)
    
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader)

    lgr.info('Train: Done', {'attrs': 'bold'})


######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS['infer_filename'] = os.path.join(PATHS["model_dir"], 'infer.gz')
INFER_COMMON_PARAMS['checkpoint'] = os.path.join(PATHS["model_dir"], "best_epoch.ckpt")


######################################
# Inference Template
######################################
def run_infer(paths: dict, infer_common_params: dict):
    #### Logger
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={infer_common_params["infer_filename"]}', {'color': 'magenta'})

    ## Data
    # Create dataset
    validation_dataset = MNIST.dataset(paths["cache_dir"], train=False)
    # dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset, collate_fn=CollateDefault(), batch_size=2, num_workers=2)

    
    pl_module = LightningModuleMnist.load_from_checkpoint(infer_common_params['checkpoint'], model_dir=paths["model_dir"], map_location="cpu", strict=True)
    pl_module.set_predictions_keys(['model.output.classification', 'data.label'])

    lgr.info('Model: Done', {'attrs': 'bold'})
    pl_trainer = pl.Trainer(default_root_dir=paths['model_dir'],
                            accelerator=TRAIN_COMMON_PARAMS["trainer.accelerator"],
                            devices=TRAIN_COMMON_PARAMS["trainer.num_devices"],
                            auto_select_gpus=True)
    
    predictions = pl_trainer.predict(pl_module, validation_dataloader, return_predictions=True)
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_common_params['infer_filename'])
    

######################################
# Analyze Common Params
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS['infer_filename'] = INFER_COMMON_PARAMS['infer_filename']


######################################
# Eval Template
######################################
def run_eval(paths: dict, eval_common_params: dict):
    create_dir(paths["eval_dir"])
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Eval', {'attrs': ['bold', 'underline']})

    # metrics
    class_names = [str(i) for i in range(10)]

    metrics = OrderedDict([
        ('operation_point', MetricApplyThresholds(pred='model.output.classification')), # will apply argmax
        ('accuracy', MetricAccuracy(pred='results:metrics.operation_point.cls_pred', target='data.label')),
        ('roc', MetricROCCurve(pred='model.output.classification', target='data.label', class_names=class_names, output_filename=os.path.join(paths['eval_dir'], 'roc_curve.png'))),
        ('auc', MetricAUCROC(pred='model.output.classification', target='data.label', class_names=class_names)),
    ])
   
    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(ids=None,
                     data=eval_common_params["infer_filename"],
                     metrics=metrics,
                     output_dir=paths['eval_dir'])

    return results


######################################
# Run
######################################
if __name__ == "__main__":
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
