import torchvision
from torchvision import transforms
from fuse.utils.utils_logger import fuse_logger_start
import logging
from fuse.data.dataset.dataset_wrapper import FuseDatasetWrapper
from fuse.data.sampler.sampler_balanced_batch import FuseSamplerBalancedBatch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torchvision.models as models
from fuse_examples.classification.mnist import lenet
from fuse.models.model_wrapper import FuseModelWrapper
from fuse_examples.classification.mnist.runner import perform_softmax
from fuse.losses.loss_default import FuseLossDefault
from typing import OrderedDict
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve, MetricConfusion
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.managers.callbacks.callback_metric_statistics import FuseMetricStatisticsCallback
from fuse.managers.callbacks.callback_tensorboard import FuseTensorboardCallback
from fuse.managers.callbacks.callback_time_statistics import FuseTimeStatisticsCallback
import torch.optim as optim
from fuse.managers.manager_default import FuseManagerDefault
import multiprocessing
from fuse.utils import gpu as FuseUtilsGPU
import os
import torch.nn.functional as F
from fuse.eval.evaluator import EvaluatorDefault 
from fuse.data.data_source.data_source_default import FuseDataSourceDefault
import copy
from fuse.models.backbones.backbone_resnet_3d import FuseBackboneResnet3D
from fuse.models.model_default import FuseModelDefault
from fuse.models.heads.head_3D_classifier import FuseHead3dClassifier
from fuse_examples.classification.knight.make_predictions_file import make_predictions_file
from fuse_examples.classification.knight.make_targets_file import make_targets_file
from fuse_examples.classification.knight.eval.eval_task1 import eval

def run_train(dataset, sample_ids, cv_index, test=False, params=None, \
        rep_index=0, rand_gen=None):
    assert(test == False)
    model_dir = os.path.join(params["paths"]["model_dir"], 'rep_' + str(rep_index), str(cv_index))
    cache_dir = os.path.join(params["paths"]["cache_dir"], 'rep_' + str(rep_index), str(cv_index))
    # obtain train/val dataset subset:
    ## Create subset data sources
    train_data_source = FuseDataSourceDefault([dataset.samples_description[i] for i in sample_ids[0]])
    validation_data_source = FuseDataSourceDefault([dataset.samples_description[i] for i in sample_ids[1]])
    ## TODO: consider if there's a better way to obtain a subset of an already created fuse dataset
    train_dataset = copy.deepcopy(dataset)
    validation_dataset = copy.deepcopy(dataset)

    print(f'- Load and cache data:')
    train_dataset.create(reset_cache=False, override_datasource=train_data_source)
    validation_dataset.create(reset_cache=False, override_datasource=validation_data_source)
    print(f'- Load and cache data: Done')

    ## Create sampler
    print(f'- Create sampler:')
    sampler = FuseSamplerBalancedBatch(dataset=train_dataset,
                                    balanced_class_name=params['common']['target_name'],
                                    num_balanced_classes=params['common']['num_classes'],
                                    batch_size=params['batch_size'],
                                    balanced_class_probs=[1.0/params['common']['num_classes']]*params['common']['num_classes'] if params['common']['task_num']==2 else None,
                                    use_dataset_cache=False) # we don't want to use_dataset_cache here since it's more 
                                                                # costly to read all cached data then simply the CSV file 
                                                                # which contains the labels

    print(f'- Create sampler: Done')

    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                shuffle=False, drop_last=False,
                                batch_sampler=sampler, collate_fn=train_dataset.collate_fn,
                                num_workers=params['num_workers'], generator=rand_gen)

    validation_dataloader = DataLoader(dataset=validation_dataset,
                                        shuffle=False,
                                        drop_last=False,
                                        batch_sampler=None,
                                        batch_size=params['batch_size'],
                                        num_workers=8,
                                        collate_fn=validation_dataset.collate_fn,
                                        generator=rand_gen)

    ## Model definition
    ##############################################################################

    if params['common']['use_data']['imaging']:
        backbone = FuseBackboneResnet3D(in_channels=1)
        conv_inputs = [('model.backbone_features', 512)]
    else:
        backbone = nn.Identity()
        conv_inputs = None
    if params['common']['use_data']['clinical']:
        append_features = [("data.input.clinical.all", 11)]
    else:
        append_features = None

    model = FuseModelDefault(
        conv_inputs=(('data.input.image', 1),),
        backbone=backbone,
        heads=[
            FuseHead3dClassifier(head_name='head_0',
                                conv_inputs=conv_inputs,
                                dropout_rate=params['imaging_dropout'], 
                                num_classes=params['common']['num_classes'],
                                append_features=append_features,
                                append_layers_description=(256,128),
                                append_dropout_rate=params['clinical_dropout'],
                                fused_dropout_rate=params['fused_dropout']
                                ),
        ]
    )

    # Loss definition:
    ##############################################################################
    losses = {
        'cls_loss': FuseLossDefault(pred_name='model.logits.head_0', target_name=params['common']['target_name'],
                                    callable=F.cross_entropy, weight=1.0)
    }

    # Metrics definition:
    ##############################################################################
    metrics = OrderedDict([
        ('op', MetricApplyThresholds(pred='model.output.head_0')), # will apply argmax
        ('auc', MetricAUCROC(pred='model.output.head_0', target=params['common']['target_name'])),
        ('accuracy', MetricAccuracy(pred='results:metrics.op.cls_pred', target=params['common']['target_name'])),
        ('sensitivity', MetricConfusion(pred='results:metrics.op.cls_pred', target=params['common']['target_name'], metrics=('sensitivity',))),

    ])

    best_epoch_source = {
        'source': params['common']['target_metric'],  # can be any key from losses or metrics dictionaries
        'optimization': 'max',  # can be either min/max
    }

    # Optimizer definition:
    ##############################################################################
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'],
                            weight_decay=0.001)         
                            
    # Scheduler definition:
    ##############################################################################
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    ## Training
    ##############################################################################

    # set tensorboard callback
    callbacks = {
        FuseTensorboardCallback(model_dir=model_dir), # save statistics for tensorboard
        FuseMetricStatisticsCallback(output_path=model_dir + "/metrics.csv"),  # save statistics a csv file

    }
    manager = FuseManagerDefault(output_model_dir=model_dir, force_reset=True)
    manager.set_objects(net=model,
                        optimizer=optimizer,
                        losses=losses,
                        metrics=metrics,
                        best_epoch_source=best_epoch_source,
                        lr_scheduler=scheduler,
                        callbacks=callbacks,
                        train_params={'num_epochs': params['num_epochs'], 'lr_sch_target': 'train.losses.total_loss'}, # 'lr_sch_target': 'validation.metrics.auc.macro_avg'
                        output_model_dir=model_dir)

    print('Training...')            
    manager.train(train_dataloader=train_dataloader,
                    validation_dataloader=validation_dataloader)


def run_infer(dataset, sample_ids, cv_index, test=False, params=None, \
              rep_index=0, rand_gen=None):

    model_dir = os.path.join(params["paths"]["model_dir"], 'rep_' + str(rep_index), str(cv_index))
    cache_dir = os.path.join(params["paths"]["cache_dir"])
    infer_dir = os.path.join(params["paths"]["inference_dir"], 'rep_' + str(rep_index), str(cv_index))

    if not os.path.isdir(infer_dir):
        os.makedirs(infer_dir)

    checkpoint = 'best'
    data_path = params['paths']['data_dir']
    predictions_filename = os.path.join(infer_dir, 'predictions.csv')
    targets_filename = os.path.join(infer_dir, 'targets.csv')

    predictions_key_name = "model.output.head_0"
    task_num = params['common']['task_num']

    split = {}
    split['train'] = [dataset.samples_description[i] for i in sample_ids[0]]
    split['val'] = [dataset.samples_description[i] for i in sample_ids[1]]

    make_predictions_file(model_dir=model_dir, checkpoint=checkpoint, data_path=data_path, cache_path=cache_dir, split=split, output_filename=predictions_filename, predictions_key_name=predictions_key_name, task_num=task_num)
    make_targets_file(data_path=data_path, cache_path=cache_dir, split=split, output_filename=targets_filename)

def run_eval(dataset, sample_ids, cv_index, test=False, params=None, \
             rep_index=0, rand_gen=None, pred_key='model.output.classification', \
             label_key='data.label'):

    infer_dir = os.path.join(params["paths"]["inference_dir"], 'rep_' + str(rep_index), str(cv_index))
    eval_dir = os.path.join(params["paths"]["eval_dir"], 'rep_' + str(rep_index), str(cv_index))
    targets_filename = os.path.join(infer_dir, 'targets.csv')
    predictions_filename = os.path.join(infer_dir, 'predictions.csv')
    output_dir = eval_dir
    eval(target_filename=targets_filename, task1_prediction_filename=predictions_filename, task2_prediction_filename="", output_dir=output_dir)


