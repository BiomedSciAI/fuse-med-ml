import torchvision
from torchvision import transforms
from fuse.utils.utils_logger import fuse_logger_start
import logging
from fuse.data.dataset.dataset_wrapper import FuseDatasetWrapper
from fuse.data.sampler.sampler_balanced_batch import FuseSamplerBalancedBatch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch
import torchvision.models as models
from fuse_examples.classification.mnist import lenet
from fuse.models.model_wrapper import FuseModelWrapper
from fuse_examples.classification.mnist.runner import perform_softmax
from fuse.losses.loss_default import FuseLossDefault
from typing import OrderedDict
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve
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

def run_train(dataset, sample_ids, cv_index, test=False, params=None, \
        rep_index=0, rand_gen=None):
    assert(test == False)
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

    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=os.path.join(params['paths']['model_dir'], 'rep_' + str(rep_index), str(cv_index)), console_verbose_level=logging.INFO, force_reset=True)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Train', {'attrs': ['bold', 'underline']})

    model_dir = os.path.join(params["paths"]["model_dir"], 'rep_' + str(rep_index), str(cv_index))
    cache_dir = os.path.join(params["paths"]["cache_dir"], 'rep_' + str(rep_index), str(cv_index))
    lgr.info(f'model_dir={model_dir}', {'color': 'magenta'})
    lgr.info(f'cache_dir={cache_dir}', {'color': 'magenta'})

    # ==============================================================================
    # Data
    # ==============================================================================

    lgr.info(f'- Create sampler:')
    sampler = FuseSamplerBalancedBatch(dataset=train_dataset,
                                       balanced_class_name='data.label',
                                       num_balanced_classes=10,
                                       batch_size=params['data.batch_size'],
                                       balanced_class_weights=None)
    lgr.info(f'- Create sampler: Done')

    # Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=sampler, num_workers=params['data.train_num_workers'], generator=rand_gen)

    # dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=params['data.batch_size'],
                                       num_workers=params['data.validation_num_workers'], generator=rand_gen)

    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info('Model:', {'attrs': 'bold'})

    if params['model'] == 'resnet18':
        torch_model = models.resnet18(num_classes=10)
        # modify conv1 to support single channel image
        torch_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif params['model'] == 'lenet':
        torch_model = lenet.LeNet()
    
    # use adaptive avg pooling to support mnist low resolution images
    torch_model.avgpool = torch.nn.AdaptiveAvgPool2d(1)

    model = FuseModelWrapper(model=torch_model,
                             model_inputs=['data.image'],
                             post_forward_processing_function=perform_softmax,
                             model_outputs=['logits.classification', 'output.classification']
                             )

    lgr.info('Model: Done', {'attrs': 'bold'})

    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        'cls_loss': FuseLossDefault(pred_name='model.logits.classification', target_name='data.label', callable=F.cross_entropy, weight=1.0),
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    metrics = OrderedDict([
        ('operation_point', MetricApplyThresholds(pred='model.output.classification')), # will apply argmax
        ('accuracy', MetricAccuracy(pred='results:metrics.operation_point.cls_pred', target='data.label'))
    ])

    # =====================================================================================
    #  Callbacks
    # =====================================================================================
    callbacks = [
        # default callbacks
        FuseTensorboardCallback(model_dir=model_dir),  # save statistics for tensorboard
        FuseMetricStatisticsCallback(output_path=model_dir + "/metrics.csv"),  # save statistics a csv file
        FuseTimeStatisticsCallback(num_epochs=params['manager.train_params']['num_epochs'], load_expected_part=0.1)  # time profiler
    ]

    # =====================================================================================
    #  Manager - Train
    # =====================================================================================
    lgr.info('Train:', {'attrs': 'bold'})

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['manager.learning_rate'], weight_decay=params['manager.weight_decay'])

    # create learning scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # train from scratch
    manager = FuseManagerDefault(output_model_dir=model_dir, force_reset=params['paths']['force_reset_model_dir'])
    # Providing the objects required for the training process.
    manager.set_objects(net=model,
                        optimizer=optimizer,
                        losses=losses,
                        metrics=metrics,
                        best_epoch_source=params['manager.best_epoch_source'],
                        lr_scheduler=scheduler,
                        callbacks=callbacks,
                        train_params=params['manager.train_params'])

    ## Continue training
    if params['manager.resume_checkpoint_filename'] is not None:
        # Loading the checkpoint including model weights, learning rate, and epoch_index.
        manager.load_checkpoint(checkpoint=params['manager.resume_checkpoint_filename'], mode='train')

    # Start training
    manager.train(train_dataloader=train_dataloader, validation_dataloader=validation_dataloader)

    lgr.info('Train: Done', {'attrs': 'bold'})


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

