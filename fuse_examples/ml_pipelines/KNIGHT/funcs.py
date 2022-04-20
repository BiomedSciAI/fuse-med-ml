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
    # obtain train/val dataset subset:
    if sample_ids is None:
        torch_validation_dataset = dataset
    else:
        torch_validation_dataset = Subset(dataset, sample_ids[1])
    # wrap torch dataset:
    validation_dataset = FuseDatasetWrapper(name='validation', dataset=torch_validation_dataset, mapping=('image', 'label'))
    validation_dataset.create()
    
    #### Logger
    model_dir = os.path.join(params['paths']['model_dir'], 'rep_' + str(rep_index), str(cv_index))
    if test:
        inference_dir = os.path.join(params['paths']['test_dir'], 'rep_' + str(rep_index), str(cv_index))
        infer_filename = params['test_infer_filename']
    else:
        inference_dir = os.path.join(params['paths']['inference_dir'], 'rep_' + str(rep_index), str(cv_index))
        infer_filename = params['infer_filename']
    fuse_logger_start(output_path=inference_dir, console_verbose_level=logging.INFO, force_reset=True)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={os.path.join(inference_dir, infer_filename)}', {'color': 'magenta'})

    # dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset, collate_fn=validation_dataset.collate_fn, batch_size=2, num_workers=2, generator=rand_gen)

    ## Manager for inference
    manager = FuseManagerDefault()
    output_columns = ['model.output.classification', 'data.label']
    manager.infer(data_loader=validation_dataloader,
                  input_model_dir=model_dir,
                  checkpoint=params['checkpoint'],
                  output_columns=output_columns,
                  output_file_name=os.path.join(inference_dir, infer_filename))


def run_eval(dataset, sample_ids, cv_index, test=False, params=None, \
             rep_index=0, rand_gen=None, pred_key='model.output.classification', \
             label_key='data.label'):
    if test:
        inference_dir = os.path.join(params['paths']['test_dir'], 'rep_' + str(rep_index), str(cv_index))
        infer_filename = params["test_infer_filename"]
    else:
        inference_dir = os.path.join(params['paths']['inference_dir'], 'rep_' + str(rep_index), str(cv_index))
        infer_filename = params["infer_filename"]
    eval_dir = os.path.join(params['paths']['eval_dir'], 'rep_' + str(rep_index), str(cv_index))
    fuse_logger_start(output_path=inference_dir, console_verbose_level=logging.INFO, force_reset=True)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Analyze', {'attrs': ['bold', 'underline']})

    # metrics
    class_names = [str(i) for i in range(10)]

    metrics = OrderedDict([
        ('operation_point', MetricApplyThresholds(pred=pred_key)), # will apply argmax
        ('accuracy', MetricAccuracy(pred='results:metrics.operation_point.cls_pred', target=label_key)),
        ('roc', MetricROCCurve(pred=pred_key, target=label_key, class_names=class_names, output_filename=os.path.join(inference_dir, 'roc_curve.png'))),
        ('auc', MetricAUCROC(pred=pred_key, target=label_key, class_names=class_names)),
    ])
   
    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    _ = evaluator.eval(ids=None,
                     data=os.path.join(inference_dir, infer_filename),
                     metrics=metrics,
                     output_dir=eval_dir)

