import logging
import random
from pathlib import Path
from glob import glob
import matplotlib.pylab as plt
import os
import numpy as np
import pandas as pd
from skimage.io import imread

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

torch.__version__

import sys
sys.path.append('Pytorch-UNet/')
sys.path.append('Pytorch-UNet/unet/')

from unet import UNet

# parameters
SZ = 512
# TRAIN = f'siim/data_bin/data{SZ}/train/'
# TEST = f'siim/data_bin/data{SZ}/test/'
# MASKS = f'siim/data_bin/data{SZ}/masks/'
TRAIN = f'siim/data{SZ}/train/'
TEST = f'siim/data{SZ}/test/'
MASKS = f'siim/data{SZ}/masks/'


def perform_softmax(output):
    if isinstance(output, torch.Tensor):  # validation
        logits = output
    else:  # train
        logits = output.logits
    cls_preds = F.softmax(logits, dim=1)
    return logits, cls_preds


def mask_size(fn):
    sz = []
    for f in fn:
        im = imread(f)
        sz.append(np.array(im>0).sum())
        # if im.sum() > 0:
        #     plt.figure()
        #     plt.imshow(im)
        #     plt.show()
    return sz


from fuse.data.dataset.dataset_default import FuseDatasetDefault
from fuse.data.augmentor.augmentor_default import FuseAugmentorDefault
from fuse.data.visualizer.visualizer_default import FuseVisualizerDefault
from fuse.data.dataset.dataset_wrapper import FuseDatasetWrapper
from fuse.data.sampler.sampler_balanced_batch import FuseSamplerBalancedBatch
from fuse.data.augmentor.augmentor_toolbox import aug_op_affine_group, aug_op_affine, aug_op_color, aug_op_gaussian, aug_op_elastic_transform
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerUniform as Uniform
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandBool as RandBool
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandInt as RandInt
from fuse.managers.callbacks.callback_tensorboard import FuseTensorboardCallback
from fuse.managers.callbacks.callback_metric_statistics import FuseMetricStatisticsCallback
from fuse.managers.callbacks.callback_time_statistics import FuseTimeStatisticsCallback
from fuse.utils.utils_logger import fuse_logger_start

# imports for training
from fuse.models.model_wrapper import FuseModelWrapper
from fuse.losses.loss_default import FuseLossDefault
from fuse.losses.segmentation.loss_dice import BinaryDiceLoss, DiceBCELoss
from fuse.losses.segmentation.loss_dice import FuseDiceLoss

# imports for validation/inference/performance 
from fuse.metrics.classification.metric_accuracy import FuseMetricAccuracy
from fuse.metrics.classification.metric_roc_curve import FuseMetricROCCurve
from fuse.metrics.classification.metric_auc import FuseMetricAUC
from fuse.analyzer.analyzer_default import FuseAnalyzerDefault
from fuse.metrics.metric_auc_per_pixel import FuseMetricAUCPerPixel
from fuse.metrics.segmentation.metric_score_map import FuseMetricScoreMap
from fuse.data.processor.processor_dataframe import FuseProcessorDataFrame

import torch.nn.functional as F
import torch.optim as optim
from fuse.managers.manager_default import FuseManagerDefault
from fuse.utils.utils_gpu import FuseUtilsGPU

from data_source_segmentation import FuseDataSourceSeg
from seg_input_processor import SegInputProcessor

# TODO: Path to save model
ROOT = ''
# TODO: path to store the data  (?? what data? after download?)
ROOT_DATA = ROOT
# TODO: Name of the experiment
EXPERIMENT = 'unet_seg_results'
# TODO: Path to cache data
CACHE_PATH = ''
# TODO: Name of the cached data folder
EXPERIMENT_CACHE = 'exp_cache'

PATHS = {'data_dir': [TRAIN, MASKS, TEST],
         'model_dir': os.path.join(ROOT, EXPERIMENT, 'model_dir'),
         'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'cache_dir': os.path.join(CACHE_PATH, EXPERIMENT_CACHE+'_cache_dir'),
         'inference_dir': os.path.join(ROOT, EXPERIMENT, 'infer_dir'),
         'analyze_dir': os.path.join(ROOT, EXPERIMENT, 'analyze_dir')}

# # augmentations from skin-fuse-code

# TRAIN_COMMON_PARAMS['data.augmentation_pipeline'] = [
#     [
#         ('data.input.input_0',),
#         aug_op_affine,
#         {'rotate': Uniform(-180.0, 180.0), 'translate': (RandInt(-50, 50), RandInt(-50, 50)),
#          'flip': (RandBool(0.3), RandBool(0.3)), 'scale': Uniform(0.9, 1.1)},
#         {'apply': RandBool(0.9)}
#     ],
#     [
#         ('data.input.input_0',),
#         aug_op_color,
#         {'add': Uniform(-0.06, 0.06), 'mul': Uniform(0.95, 1.05), 'gamma': Uniform(0.9, 1.1),
#          'contrast': Uniform(0.85, 1.15)},
#         {'apply': RandBool(0.7)}
#     ],
#     [
#         ('data.input.input_0',),
#         aug_op_gaussian,
#         {'std': 0.03},
#         {'apply': RandBool(0.7)}
#     ],
# ]

##########################################
# Train Common Params
##########################################
# ============
# Data
# ============
TRAIN_COMMON_PARAMS = {}
TRAIN_COMMON_PARAMS['data.batch_size'] = 32
TRAIN_COMMON_PARAMS['data.train_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.augmentation_pipeline'] = [
    # TODO: define the augmentation pipeline here
    # Fuse TIP: Use as a reference the simple augmentation pipeline written in Fuse.data.augmentor.augmentor_toolbox.aug_image_default_pipeline
    [
        ('data.input.input_0','data.gt.gt_global'),
        aug_op_affine_group,
        {'rotate': Uniform(-20.0, 20.0),  # Uniform(-20.0, 20.0),
        'flip': (RandBool(0.0), RandBool(0.5)),  # (RandBool(1.0), RandBool(0.5)),
        'scale': Uniform(0.9, 1.1),
        'translate': (RandInt(-50, 50), RandInt(-50, 50))},
        {'apply': RandBool(0.9)}
    ],
    [
        ('data.input.input_0','data.gt.gt_global'),
        aug_op_elastic_transform,
        {},
        {'apply': RandBool(0.7)}
    ],
    [
        ('data.input.input_0',),
        aug_op_color,
        {
         'add': Uniform(-0.06, 0.06), 
         'mul': Uniform(0.95, 1.05), 
         'gamma': Uniform(0.9, 1.1),
         'contrast': Uniform(0.85, 1.15)
        },
        {'apply': RandBool(0.7)}
    ],
    [
        ('data.input.input_0',),
        aug_op_gaussian,
        {'std': 0.05},
        {'apply': RandBool(0.7)}
    ],
]
# ===============
# Manager - Train1
# ===============
TRAIN_COMMON_PARAMS['manager.train_params'] = {
    'num_epochs': 200,

    'virtual_batch_size': 1,  # number of batches in one virtual batch
    'start_saving_epochs': 10,  # first epoch to start saving checkpoints from
    'gap_between_saving_epochs': 5,  # number of epochs between saved checkpoint
}
TRAIN_COMMON_PARAMS['manager.best_epoch_source'] = {
    'source': 'losses.total_loss',  # can be any key from 'epoch_results' (either metrics or losses result)
    'optimization': 'min',  # can be either min/max
    'on_equal_values': 'better',  ## ?? why is it important??
    # can be either better/worse - whether to consider best epoch when values are equal
}
TRAIN_COMMON_PARAMS['manager.learning_rate'] = 1e-1
TRAIN_COMMON_PARAMS['manager.weight_decay'] = 1e-4  # 0.001
TRAIN_COMMON_PARAMS['manager.resume_checkpoint_filename'] = None  # if not None, will try to load the checkpoint
## Give a default checkpoint name? load a default checkpoint?

# allocate gpus
NUM_GPUS = 4
if NUM_GPUS == 0:
    TRAIN_COMMON_PARAMS['manager.train_params']['device'] = 'cpu' 

TRAIN_COMMON_PARAMS['partition_file'] = 'train_val_split.pickle'
TRAIN_COMMON_PARAMS['manager.train'] = False  # if not None, will try to load the checkpoint
# ==================================================================
def vis_batch(sample, num_rows=3):
    img_names = sample['data']['descriptor'][0]
    mask_names = sample['data']['descriptor'][1]
    
    img = sample['data']['input']['input_0']
    mask = sample['data']['gt']['gt_global']
    
    n = img.shape[0]
    num_col = n // num_rows + 1
    fig, ax = plt.subplots(num_rows, num_col, figsize=(14, 3*num_rows))
    ax = ax.ravel()
    for i in range(n):
        im = img[i].squeeze()
        msk = mask[i].squeeze()
        
        if im.shape[0] == 3:
            im = im.permute((1,2,0))  # im is a tensor

        ax[i].imshow(im,cmap='bone')
        ax[i].imshow(msk,alpha=0.5,cmap='Reds')        
        #         ax[i, 1].imshow(msk)


def main(paths: dict, train_common_params: dict, train=True, infer=True):

    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    FuseUtilsGPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    train_common_params['manager.train'] = train  # if not None, will try to load the checkpoint
    train_common_params['manager.infer'] = infer

    train_path = paths['data_dir'][0]
    mask_path = paths['data_dir'][1]
    test_path =paths['data_dir'][2]

    train_fn = glob(train_path + '/*')
    train_fn.sort()

    masks_fn = glob(mask_path + '/*')
    masks_fn.sort()

    m_size = mask_size(masks_fn)
    size_inx = np.argsort(m_size)

    # train_fn = np.array(train_fn)[size_inx[-200:]]
    # masks_fn = np.array(masks_fn)[size_inx[-200:]]

    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)

    # fn = list(zip(train_fn, masks_fn))

    # # split to train-validation -
    # VAL_SPLIT = 0.2  # frac of validation set
    # n_train = int(len(fn) * (1-VAL_SPLIT))

    # # random shuffle the file-list
    # # random.shuffle(fn)
    # train_fn = fn[:n_train]
    # train_size = m_size[:n_train]
    # val_fn = fn[n_train:]

    # size_inx = np.argsort(train_size)
    # train_fn = np.array(train_fn)[size_inx[-3000:]].tolist()
    # train_fn = [tuple(tr) for tr in train_fn]

    # # filter only train samples with positive mask
    # train_fn = np.array(train_fn)[np.array(train_size) > 0].tolist()
    # train_fn = [tuple(tr) for tr in train_fn]

    train_data_source = FuseDataSourceSeg(image_source=train_path,
                                          mask_source=mask_path,
                                          partition_file=train_common_params['partition_file'],
                                          train=True)
    # train_data_source = FuseDataSourceSeg(train_fn)
    print(train_data_source.summary())

    ## Create data processors:
    input_processors = {
        'input_0': SegInputProcessor(name='image')
    }
    gt_processors = {
        'gt_global': SegInputProcessor(name='mask')
    }

    ## Create data augmentation (optional)
    augmentor = FuseAugmentorDefault(augmentation_pipeline=train_common_params['data.augmentation_pipeline'])

    # Create visualizer (optional)
    visualiser = FuseVisualizerDefault(image_name='data.input.input_0', 
                                       mask_name='data.gt.gt_global',
                                       pred_name='model.logits.classification')

    train_dataset = FuseDatasetDefault(cache_dest=None,
                                    data_source=train_data_source,
                                    input_processors=input_processors,
                                    gt_processors=gt_processors,
                                    augmentor=augmentor,
                                    visualizer=visualiser)
    train_dataset.create()

    # debug_size = []
    # for data in train_dataset:
    #     img = data['data']['input']['input_0'].numpy().squeeze()
    #     mask = data['data']['gt']['gt_global'].numpy().squeeze()
    #     # if mask.sum() > 0:
    #     debug_size.append(mask.sum())

    # ==================================================================
    # Validation dataset
    valid_data_source = FuseDataSourceSeg(image_source=train_path,
                                          mask_source=mask_path,
                                          partition_file=train_common_params['partition_file'],
                                          train=False)
    print(valid_data_source.summary())
    # valid_data_source = FuseDataSourceSeg(val_fn)
    # valid_data_source.summary()

    valid_dataset = FuseDatasetDefault(cache_dest=None,
                                    data_source=valid_data_source,
                                    input_processors=input_processors,
                                    gt_processors=gt_processors,
                                    visualizer=visualiser)
    valid_dataset.create()

    ## Create sampler
    # sampler = FuseSamplerBalancedBatch(dataset=train_dataset,
    #                                    balanced_class_name='data.gt.gt_global.tensor',
    #                                    num_balanced_classes=2,
    #                                    batch_size=train_common_params['data.batch_size'])


    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=True, 
                                  drop_last=False,
                                  batch_size=train_common_params['data.batch_size'],
                                  collate_fn=train_dataset.collate_fn,
                                  num_workers=train_common_params['data.train_num_workers'])
    #                               batch_sampler=sampler, collate_fn=train_dataset.collate_fn,
    #                               num_workers=train_common_params['data.train_num_workers'])

    ## Create dataloader
    validation_dataloader = DataLoader(dataset=valid_dataset,
                                       shuffle=False, 
                                       drop_last=False,
                                       batch_size=train_common_params['data.batch_size'],
                                       collate_fn=train_dataset.collate_fn,
                                       num_workers=train_common_params['data.validation_num_workers'])

    if False:
        # train_dataset.visualize(10)

        inx = 10 #2405
        data = train_dataset.get(inx)
        img = data['data']['input']['input_0'].numpy().squeeze()
        mask = data['data']['gt']['gt_global'].numpy().squeeze()

        data = train_dataset.getitem_without_augmentation(inx)
        img_aug = data['data']['input']['input_0'].numpy().squeeze()
        mask_aug = data['data']['gt']['gt_global'].numpy().squeeze()

        if img.shape[0] == 3:
            img = img.transpose((1,2,0))
            img_aug = img_aug.transpose((1,2,0))

        fig, axs = plt.subplots(1,2, figsize=(14,7))
        axs[0].imshow(img, plt.cm.bone)
        axs[0].imshow(1-mask, 'hot', alpha=0.4)
        axs[1].imshow(img_aug, plt.cm.bone)
        axs[1].imshow(1-mask_aug, 'hot', alpha=0.4)
        # axs[1].imshow(mask, interpolation=None)
        plt.show()

        print('Num of positive pixels - ', mask.sum())

        i = 0
        for batch in train_dataloader:
            vis_batch(batch)
            i += 1
            if i > 10:
                break
        plt.show()
    # ==================================================================

    # # Training graph
    torch_model = UNet(n_channels=1, n_classes=1, bilinear=False)
    net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)

    import ipdb; ipdb.set_trace(context=7) # BREAKPOINT
    model = FuseModelWrapper(model=torch_model,
                            model_inputs=['data.input.input_0'],
                            post_forward_processing_function=perform_softmax,
                            model_outputs=['logits.classification', 'output.classification']
                            )

    # # take one batch:
    # batch = next(iter(train_dataloader))
    # img = batch['data']['input']['input_0']
    # img.shape

    # pred_mask = torch_model(img)
    # pred_mask.shape

    # ====================================================================================
    #  Loss
    # ====================================================================================
    # dice_loss = BinaryDiceLoss()
    dice_loss = DiceBCELoss()
    # losses = {
    #     'dice_loss': FuseDiceLoss(pred_name='model.logits.classification', 
    #                                 target_name='data.gt.gt_global')
    # }
    losses = {
        'cls_loss': FuseLossDefault(pred_name='model.logits.classification', 
                                    target_name='data.gt.gt_global',
                                    callable=dice_loss, 
                                    weight=1.0)
    }

    model = model.cuda()
    # create optimizer
    # optimizer = optim.AdamW(model.parameters(), 
    #                        lr=train_common_params['manager.learning_rate'],
    #                        weight_decay=train_common_params['manager.weight_decay'])
    optimizer = optim.SGD(model.parameters(), 
                          lr=train_common_params['manager.learning_rate'],
                          momentum=0.9,
                          weight_decay=train_common_params['manager.weight_decay'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # train from scratch
    if train_common_params['manager.train']:
        manager = FuseManagerDefault(output_model_dir=paths['model_dir'], 
                                    force_reset=paths['force_reset_model_dir'])
    else:
        manager = FuseManagerDefault()

    # =====================================================================================
    #  Callbacks
    # =====================================================================================
    callbacks = [
        # default callbacks
        FuseTensorboardCallback(model_dir=paths['model_dir']),  # save statistics for tensorboard
        FuseMetricStatisticsCallback(output_path=paths['model_dir'] + "/metrics.csv"),  # save statistics in a csv file
        FuseTimeStatisticsCallback(num_epochs=train_common_params['manager.train_params']['num_epochs'], load_expected_part=0.1)  # time profiler
    ]

    # Providing the objects required for the training process.
    manager.set_objects(net=model,
                        optimizer=optimizer,
                        losses=losses,
                        lr_scheduler=scheduler,
                        callbacks=callbacks,
                        best_epoch_source=train_common_params['manager.best_epoch_source'],
                        train_params=train_common_params['manager.train_params'],
                        output_model_dir=paths['model_dir'])


    # Start training
    if train_common_params['manager.train']:
        manager.train(train_dataloader=train_dataloader,
                    validation_dataloader=validation_dataloader)

        # plot the training process:
        csv_file = os.path.join(paths['model_dir'], 'metrics.csv')
        metrics = pd.read_csv(csv_file)
        metrics.drop(index=metrics.index[0], axis=0, inplace=True)  # remove the 1st validation run

        epochs = metrics[metrics['mode'] == 'validation']['epoch']
        loss_key = 'losses.' + list(losses.keys())[0]
        val_loss = metrics[metrics['mode'] == 'validation'][loss_key]
        train_loss = metrics[metrics['mode'] == 'train'][loss_key]
 
        plt.figure()
        plt.plot(epochs, val_loss, '.-', label='validation')
        plt.plot(epochs, train_loss, '.-', label='train')
        plt.legend()
        plt.title('train and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.savefig(os.path.join(paths['model_dir'], 'train_progress.png'))
        plt.close()
        
    ################################################################################
    # Inference
    ################################################################################

    if train_common_params['manager.infer']:
        ######################################
        # Inference Common Params
        ######################################
        INFER_COMMON_PARAMS = {}
        INFER_COMMON_PARAMS['infer_filename'] = os.path.join(PATHS['inference_dir'], 'validation_set_infer.gz')
        INFER_COMMON_PARAMS['checkpoint'] = 'best' #'best'  # Fuse TIP: possible values are 'best', 'last' or epoch_index.
        output_columns = ['model.logits.classification', 'data.gt.gt_global']
        infer_common_params = INFER_COMMON_PARAMS

        manager.load_checkpoint(infer_common_params['checkpoint'], 
                                model_dir=paths['model_dir'])
        print('Skip training ...')

        manager.infer(data_loader=validation_dataloader,
                    input_model_dir=paths['model_dir'],
                    output_columns=output_columns,
                    output_file_name=infer_common_params['infer_filename'])  #,
                    # num_workers=0) 

        # visualize the predictions
        infer_processor = FuseProcessorDataFrame(data_pickle_filename=infer_common_params['infer_filename'])
        descriptors_list = infer_processor.get_samples_descriptors()
        out_name = 'model.logits.classification'
        gt_name = 'data.gt.gt_global' 
        for desc in descriptors_list[:10]:
            data = infer_processor(desc)
            pred = np.squeeze(data[out_name])
            gt = np.squeeze(data[gt_name])
            _, ax = plt.subplots(1,2)
            ax[0].imshow(pred)
            ax[0].set_title('prediction')
            ax[1].imshow(gt)
            ax[1].set_title('gt')
        plt.show()

        ######################################
        # Analyze Common Params
        ######################################
        ANALYZE_COMMON_PARAMS = {}
        ANALYZE_COMMON_PARAMS['infer_filename'] = INFER_COMMON_PARAMS['infer_filename']
        ANALYZE_COMMON_PARAMS['output_filename'] = os.path.join(PATHS['analyze_dir'], 'all_metrics')
        analyze_common_params = ANALYZE_COMMON_PARAMS

        # metrics
        metrics = {
            # 'accuracy': FuseMetricAccuracy(pred_name='model.logits.classification', target_name='data.gt.gt_global'),
            # 'roc': FuseMetricROCCurve(pred_name='model.logits.classification', target_name='data.gt.gt_global', output_filename='roc_curve.png'),
            # 'auc': FuseMetricAUC(pred_name='model.logits.classification', target_name='data.gt.gt_global')
            'auc': FuseMetricAUCPerPixel(pred_name='model.logits.classification', 
                                        target_name='data.gt.gt_global', 
                                        output_filename='roc_curve.png'),
            'seg': FuseMetricScoreMap(pred_name='model.logits.classification', 
                target_name='data.gt.gt_global',
                hard_threshold=True, threshold=0.5)
        }

        # manager.visualize(visualizer=visualiser,
        #                   data_loader=validation_dataloader, device='cpu')
                # descriptors=<optional - list of descriptors>,
                # display_func=<optional - display_func>,
                # infer_processor=None)

        # create analyzer
        analyzer = FuseAnalyzerDefault()

        # run
        # FIXME: simplify analyze interface for this case
        analyzer.analyze(gt_processors=gt_processors,
                        data_pickle_filename=analyze_common_params['infer_filename'],
                        metrics=metrics,
                        print_results=True,
                        output_filename=analyze_common_params['output_filename'],
                        num_workers=0) 


if __name__ == '__main__':
    import argparse

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--train', action='store_true')
    my_parser.add_argument('--infer', action='store_true')
    args = my_parser.parse_args()

    print(vars(args))
    params = vars(args)
    main(PATHS, TRAIN_COMMON_PARAMS, train=params['train'], infer=params['infer'])
