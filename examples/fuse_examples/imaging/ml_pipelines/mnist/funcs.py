import copy
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

from fuse.utils.utils_logger import fuse_logger_start
import logging
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.lightning.pl_module import LightningModuleDefault

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
from fuseimg.datasets.mnist import MNIST

def create_dataset(cache_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Train dataset:
    torch_train_dataset = torchvision.datasets.MNIST(cache_dir, download=True, train=True, transform=transform)

    # Validation dataset:
    torch_test_dataset = torchvision.datasets.MNIST(cache_dir, download=True, train=False, transform=transform)

    return torch_train_dataset, torch_test_dataset

def create_model() -> torch.nn.Module:
    torch_model = lenet.LeNet()
    # wrap basic torch model to automatically read inputs from batch_dict and save its outputs to batch_dict
    model = ModelWrapSeqToDict(
        model=torch_model,
        model_inputs=["data.image"],
        post_forward_processing_function=perform_softmax,
        model_outputs=["logits.classification", "output.classification"],
    )
    return model

def run_train(dataset, sample_ids, cv_index, test=False, params=None, \
        rep_index=0, rand_gen=None):
    assert(test == False)
    # obtain train/val dataset subset:
    train_dataset = Subset(dataset, sample_ids[0])
    validation_dataset = Subset(dataset, sample_ids[1])

    model_dir = os.path.join(params["paths"]["model_dir"], 'rep_' + str(rep_index), str(cv_index))
    cache_dir = os.path.join(params["paths"]["cache_dir"], 'rep_' + str(rep_index), str(cv_index))

    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=model_dir, console_verbose_level=logging.INFO)
    print("Fuse Train")

    # ==============================================================================
    # Data
    # ==============================================================================
    ## Train data
    print("- Create sampler:")
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.label",
        num_balanced_classes=10,
        batch_size=params['data.batch_size'],
        balanced_class_weights=None,
    )
    print("- Create sampler: Done")

    # Create dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=params['data.train_num_workers'],
    )
    print("Data - trainset: Done")

    ## Validation data
    # dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=params['data.batch_size'],
        collate_fn=CollateDefault(),
        num_workers=params['data.validation_num_workers'],
    )
    print("Data - validation set: Done")

    # ====================================================================================
    # Model
    # ====================================================================================
    model = create_model()

    # ====================================================================================
    # Losses
    # ====================================================================================
    losses = {
        "cls_loss": LossDefault(
            pred="model.logits.classification", target="data.label", callable=F.cross_entropy, weight=1.0
        ),
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    train_metrics = OrderedDict(
        [
            ("operation_point", MetricApplyThresholds(pred="model.output.classification")),  # will apply argmax
            ("accuracy", MetricAccuracy(pred="results:metrics.operation_point.cls_pred", target="data.label")),
        ]
    )

    validation_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
    best_epoch_source = dict(
        monitor="validation.metrics.accuracy",
        mode="max",
    )

    # ====================================================================================
    # Training components
    # ====================================================================================
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['opt.learning_rate'], weight_decay=params["opt.weight_decay"])

    # create learning scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")

    # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    # ====================================================================================
    # Train
    # ====================================================================================
    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=model_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )

    # create lightining trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=model_dir,
        max_epochs=params["trainer.num_epochs"],
        accelerator=params["trainer.accelerator"],
        strategy=params["trainer.strategy"],
        devices=params["trainer.num_devices"],
        auto_select_gpus=True,
    )

    # train
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader, ckpt_path=train_params["trainer.ckpt_path"])
    print("Train: Done")




    ##### end fuse2 code














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
    if cv_index=='ensemble':
        infer_filename = 'ensemble_results.gz'
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
                     output_dir=inference_dir)

