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
from fuse.utils.utils_gpu import FuseUtilsGPU

def create_dataset(params):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Train dataset:
    torch_train_dataset = torchvision.datasets.MNIST(params['common']['paths']['cache_dir'], download=True, train=True, transform=transform)
    train_dataset = FuseDatasetWrapper(name='train', dataset=torch_train_dataset, mapping=('image', 'label'))
    train_dataset.create()

    # Validation dataset:
    torch_test_dataset = torchvision.datasets.MNIST(params['common']['paths']['cache_dir'], download=True, train=False, transform=transform)
    # wrapping torch dataset
    test_dataset = FuseDatasetWrapper(name='test', dataset=torch_test_dataset, mapping=('image', 'label'))
    test_dataset.create()
    return train_dataset, test_dataset

def run_train(params, dataset, available_gpu_ids, sample_ids, cv_index):

    ## choose gpu id for this process
    #cpu_name = multiprocessing.current_process().name
    #cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
    #gpu_id = available_gpu_ids[cpu_id]
    #FuseUtilsGPU.choose_and_enable_multiple_gpus(1, force_gpus=[gpu_id])

    # obtain train/val dataset subset:
    train_dataset = Subset(dataset, sample_ids[0])
    validation_dataset = Subset(dataset, sample_ids[1])

    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=params['common']['paths']['model_dir'], console_verbose_level=logging.INFO, force_reset=True)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={params["common"]["paths"]["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={params["common"]["paths"]["cache_dir"]}', {'color': 'magenta'})

    # ==============================================================================
    # Data
    # ==============================================================================

    lgr.info(f'- Create sampler:')
    sampler = FuseSamplerBalancedBatch(dataset=train_dataset,
                                       balanced_class_name='data.label',
                                       num_balanced_classes=10,
                                       batch_size=params['train']['data.batch_size'],
                                       balanced_class_weights=None)
    lgr.info(f'- Create sampler: Done')

    # Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=sampler, num_workers=params['train']['data.train_num_workers'])

    # dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=params['train']['data.batch_size'],
                                       num_workers=params['train']['data.validation_num_workers'])

    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info('Model:', {'attrs': 'bold'})

    if params['train']['model'] == 'resnet18':
        torch_model = models.resnet18(num_classes=10)
        # modify conv1 to support single channel image
        torch_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif params['train']['model'] == 'lenet':
        torch_model = lenet.Net()
    
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
        FuseTensorboardCallback(model_dir=params['common']['paths']['model_dir']),  # save statistics for tensorboard
        FuseMetricStatisticsCallback(output_path=params['common']['paths']['model_dir'] + "/metrics.csv"),  # save statistics a csv file
        FuseTimeStatisticsCallback(num_epochs=params['train']['manager.train_params']['num_epochs'], load_expected_part=0.1)  # time profiler
    ]

    # =====================================================================================
    #  Manager - Train
    # =====================================================================================
    lgr.info('Train:', {'attrs': 'bold'})

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['train']['manager.learning_rate'], weight_decay=params['train']['manager.weight_decay'])

    # create learning scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # train from scratch
    manager = FuseManagerDefault(output_model_dir=params['common']['paths']['model_dir'], force_reset=params['common']['paths']['force_reset_model_dir'])
    # Providing the objects required for the training process.
    manager.set_objects(net=model,
                        optimizer=optimizer,
                        losses=losses,
                        metrics=metrics,
                        best_epoch_source=params['train']['manager.best_epoch_source'],
                        lr_scheduler=scheduler,
                        callbacks=callbacks,
                        train_params=params['train']['manager.train_params'])

    ## Continue training
    if params['train']['manager.resume_checkpoint_filename'] is not None:
        # Loading the checkpoint including model weights, learning rate, and epoch_index.
        manager.load_checkpoint(checkpoint=params['train']['manager.resume_checkpoint_filename'], mode='train')

    # Start training
    manager.train(train_dataloader=train_dataloader, validation_dataloader=validation_dataloader)

    lgr.info('Train: Done', {'attrs': 'bold'})


def run_eval():
    pass

def run_infer():
    pass

