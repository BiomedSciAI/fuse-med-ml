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

from fuse_examples.imaging.classification.ukbb_prostate import cohort_and_label_def, files_download_from_cos

import sys
import copy
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
import pandas as pd
from fuse.utils.utils_debug import FuseDebug
from fuse.utils.gpu import choose_and_enable_multiple_gpus

import logging

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils import NDict
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.split import dataset_balanced_division_to_folds
from fuse.dl.losses.loss_default import LossDefault

from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC, MetricAccuracy
from fuseimg.datasets.ukbb_neck_to_knee import UKBB
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe
from pytorch_lightning import Trainer
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.utils.file_io.file_io import create_dir, load_pickle, save_dataframe
from fuse.eval.evaluator import EvaluatorDefault
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from fuse.dl.models.backbones.backbone_resnet_3d import BackboneResnet3D
from fuse.dl.models import ModelMultiHead
from fuse.dl.models.heads.head_3D_classifier import Head3DClassifier

from fuse.dl.models.model_wrapper import ModelWrapDictToSeq
from medcam import medcam
import numpy as np
from cv2 import cv2
import nibabel as nib

assert "UKBB_MRI_BODY_DATA_PATH" in os.environ, "Expecting environment variable UKBB_MRI_BODY_DATA_PATH to be set. Follow the instruction in example README file to download and set the path to the data"
##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)


def create_model(train: NDict, paths: NDict) -> torch.nn.Module:
    """ 
    creates the model 
    See HeadGlobalPoolingClassifier for details 
    """
    #### Train Data
    gt_label_key = "data.gt.classification"
    class_names = cohort_and_label_def.get_class_names(train['target'])
    num_classes = len(class_names)
    model = ModelMultiHead(
        conv_inputs=(('data.input.img', 1),),
        backbone=BackboneResnet3D(in_channels=1),
        heads=[
            Head3DClassifier(head_name='head_0',
                             conv_inputs=[("model.backbone_features", 512)],
                             #  dropout_rate=train_params['imaging_dropout'],
                             #  append_dropout_rate=train_params['clinical_dropout'],
                             #  fused_dropout_rate=train_params['fused_dropout'],
                             num_classes=num_classes,
                             #  append_features=[("data.input.clinical", 8)],
                             #  append_layers_description=(256,128),
                             ),
        ])
    # create lightining trainer.
    pl_trainer = Trainer(default_root_dir=paths['model_dir'],
                         max_epochs=train['trainer']['num_epochs'],
                         accelerator=train['trainer']['accelerator'],
                         devices=train['trainer']['devices'],
                         num_sanity_val_steps=-1,
                         resume_from_checkpoint=train.get('resume_from_checkpoint'),
                         auto_select_gpus=True)
    return model, pl_trainer, num_classes, gt_label_key, class_names


#################################
# Train Template
#################################
def run_train(paths: NDict, train: NDict) -> torch.nn.Module:
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO, list_of_source_files=['../conf/config.yaml'])
    lgr = logging.getLogger('Fuse')

    clinical_data_df, var_namespace = read_clinical_data_file(filename=paths["clinical_data_file"], target=train['target'],
                                                              columns_to_add=train.get('columns_to_add'),
                                                              return_var_namespace=True)

    sample_ids = cohort_and_label_def.get_samples_for_cohort(cohort_config=train['cohort'], var_namespace=var_namespace, lgr=lgr)

    # Download data
    # instructions how to get the ukbb data
    # 1. apply for access in his website https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access
    # 2. download all data to the path configured in os env variable UKBB_DATA_PATH
    files_download_from_cos.download_sample_files(sample_ids=sample_ids, mri_output_dir=paths["data_dir"], cos_cfg=train["cos"])

    lgr.info('\nFuse Train', {'attrs': ['bold', 'underline']})
    lgr.info('cohort def=' + str(train['cohort']), {'color': 'magenta'})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})

    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})
    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info('Model:', {'attrs': 'bold'})
    model, pl_trainer, num_classes, gt_label, class_names = create_model(train, paths)
    lgr.info('Model: Done', {'attrs': 'bold'})

    # split to folds randomly - temp

    # samples_path = os.path.join(paths["data_misc_dir"],"samples.csv")
    # if os.path.isfile(samples_path) :
    #     sample_ids = pd.read_csv(samples_path)['file'].to_list()
    #     print(sample_ids)
    # else:
    #     sample_ids = None

    dataset_all = UKBB.dataset(data_dir=paths["data_dir"], target=train['target'], series_config=train['series_config'],
                               input_source_gt=clinical_data_df, cache_dir=paths["cache_dir"],
                               reset_cache=False, num_workers=train["num_workers"], sample_ids=sample_ids,
                               train=True
                               )
    print("dataset size", len(dataset_all))

    folds = dataset_balanced_division_to_folds(dataset=dataset_all,
                                               output_split_filename=os.path.join(paths["data_misc_dir"], paths["data_split_filename"]),
                                               id='data.patientID',
                                               keys_to_balance=[gt_label],
                                               nfolds=train["num_folds"],
                                               workers=train["num_workers"])

    train_sample_ids = []
    for fold in train["train_folds"]:
        train_sample_ids += folds[fold]
    validation_sample_ids = []
    for fold in train["validation_folds"]:
        validation_sample_ids += folds[fold]

    train_dataset = UKBB.dataset(data_dir=paths["data_dir"], target=train['target'], series_config=train['series_config'],
                                 input_source_gt=clinical_data_df, cache_dir=paths["cache_dir"], reset_cache=False, num_workers=train["num_workers"],
                                 sample_ids=train_sample_ids, train=True)

    validation_dataset = UKBB.dataset(data_dir=paths["data_dir"], target=train['target'], series_config=train['series_config'],
                                      input_source_gt=clinical_data_df, cache_dir=paths["cache_dir"], reset_cache=False,
                                      num_workers=train["num_workers"], sample_ids=validation_sample_ids)

    ## Create sampler
    lgr.info(f'- Create sampler:')
    sampler = BatchSamplerDefault(dataset=train_dataset,
                                  balanced_class_name=gt_label,
                                  num_balanced_classes=num_classes,
                                  batch_size=train["batch_size"],
                                  mode="approx",
                                  workers=train["num_workers"],
                                  balanced_class_weights=None
                                  )

    lgr.info(f'- Create sampler: Done')

    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=False, drop_last=False,
                                  batch_sampler=sampler, collate_fn=CollateDefault(),
                                  num_workers=train["num_workers"])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})

    #### Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})

    ## Create dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       shuffle=False,
                                       drop_last=False,
                                       batch_sampler=None,
                                       batch_size=train["batch_size"],
                                       num_workers=train["num_workers"],
                                       collate_fn=CollateDefault())
    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})
    # ===================================================================

    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        'cls_loss': LossDefault(pred='model.logits.head_0', target=gt_label,
                                callable=F.cross_entropy, weight=1.0)
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    train_metrics = OrderedDict([
        ('op', MetricApplyThresholds(pred='model.output.head_0')),  # will apply argmax
        ('auc', MetricAUCROC(pred='model.output.head_0', target=gt_label, class_names=class_names)),
        ('accuracy', MetricAccuracy(pred='results:metrics.op.cls_pred', target=gt_label)),
    ])

    validation_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric). 
    best_epoch_source = dict(
        monitor="validation.metrics.auc.macro_avg",
        mode="max",
    )

    # =====================================================================================
    #  Manager - Train
    #  Create a manager, training objects and run a training process.
    # =====================================================================================
    lgr.info('Train:', {'attrs': 'bold'})

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=train["learning_rate"],
                           weight_decay=train["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    lr_sch_config = dict(scheduler=scheduler, monitor="validation.losses.total_loss")

    # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    # create instance of PL module - FuseMedML generic version

    pl_module = LightningModuleDefault(model_dir=paths["model_dir"],
                                       model=model,
                                       losses=losses,
                                       train_metrics=train_metrics,
                                       validation_metrics=validation_metrics,
                                       best_epoch_source=best_epoch_source,
                                       optimizers_and_lr_schs=optimizers_and_lr_schs)

    # train from scratch
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader, ckpt_path=train['trainer']['ckpt_path'])
    lgr.info('Train: Done', {'attrs': 'bold'})

    return model, pl_trainer


######################################
# Inference Template
######################################
def run_infer(train: NDict, paths: NDict, infer: NDict):
    create_dir(paths['inference_dir'])
    #### Logger
    fuse_logger_start(output_path=paths["inference_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})

    pl_module, pl_trainer, infer_dataloader = load_model_and_test_data(train, paths, infer)

    infer_file = os.path.join(paths['inference_dir'], infer['infer_filename'])

    pl_module.set_predictions_keys(['model.output.head_0', 'data.gt.classification'])  # which keys to extract and dump into file
    # create a trainer instance
    predictions = pl_trainer.predict(pl_module, infer_dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)


######################################
# Explain Template
######################################
def run_explain(train: NDict, paths: NDict, infer: NDict):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Explain', {'attrs': ['bold', 'underline']})

    pl_module, _, infer_dataloader = load_model_and_test_data(train, paths, infer)

    model = ModelWrapDictToSeq(pl_module._model, output_key='head_0')
    model = medcam.inject(model, output_dir="attention_maps", backend='gcam', save_maps=True, layer='auto', return_attention=True)
    for i, batch in enumerate(infer_dataloader):
        logit, attention_map = model(batch['data.input.img'], batch['data.gt.classification'])
        attention_map = attention_map[0][0].numpy()
        batch['data.input.img'] = batch['data.input.img'][0][0].numpy()
        original_attention_map = nib.load(
            os.path.join('attention_maps', 'model.backbone.layer4', 'attention_map_' + str(i) + '_0_0.nii.gz')).get_fdata()
        original_transposed = np.transpose(batch['data.input.img'], axes=(1, 2, 0))
        scale_ratio = [original_transposed.shape[i] / value for i, value in enumerate(original_attention_map.shape)]
        points = []
        max_value = original_attention_map.argmax()
        current_max = max_value
        while True:
            current_max = original_attention_map.argmax()
            max_volume = np.unravel_index(current_max, original_attention_map.shape)
            if current_max < max_value:
                break
            points.append(np.asarray(max_volume))
            original_attention_map[max_volume] = 0.0
        print("sample ", i)
        print("points", points)
        max_volume = points[0]
        center = [int(index / 2) for index in original_attention_map.shape]
        min_dist = 99999999999999999
        for point in points:
            dist = np.linalg.norm(point - center)
            if dist < min_dist:
                min_dist = dist
                max_volume = point
        print("best", max_volume)
        attention_map = show_attention_on_image(batch['data.input.img'], attention_map)
        batch['data.input.img'] = np.transpose(batch['data.input.img'], axes=(1, 2, 0))
        original = nib.Nifti1Image(batch['data.input.img'], affine=np.eye(4))
        volume_box = np.zeros(batch['data.input.img'].shape)
        for point in points:
            bouding_box_indices = [(int((point[i] - 1) * scale_ratio[i]), int((point[i] + 1) * scale_ratio[i])) for i in range(3)]
            print(bouding_box_indices)
            volume_box = draw_bbox_around_volume(volume_box, bouding_box_indices, 1)
        # bouding_box_indices =  [(int((max_volume[i]-1)*scale_ratio[i]),int((max_volume[i]+1)*scale_ratio[i])) for i in range(3)]
        # volume_box = draw_bbox_around_volume(volume_box,max_volume, 2)
        volume_box = nib.Nifti1Image(volume_box, affine=np.eye(4))
        nib.save(original, filename=os.path.join('attention_maps', 'original_' + str(i) + '_' + batch['data.input.img_path'][0] + '_label_=' + str(
            batch['data.gt.classification']) + '.nii.gz'))
        nib.save(volume_box, filename=os.path.join('attention_maps', 'maxvolume_' + str(i) + '_' + batch['data.input.img_path'][0] + '_label_=' + str(
            batch['data.gt.classification']) + '.nii.gz'))
        nib.save(attention_map, filename=os.path.join('attention_maps',
                                                      'attention_' + str(i) + '_' + batch['data.input.img_path'][0] + '_label_=' + str(
                                                          batch['data.gt.classification']) + '.nii.gz'))


def draw_bbox_around_volume(volume_box, bouding_box_indices, color):
    for slice in range(bouding_box_indices[2][0], bouding_box_indices[2][1] - 1):
        for x in range(bouding_box_indices[0][0], bouding_box_indices[0][1] - 1):
            volume_box[x, bouding_box_indices[1][0], slice] = color
            volume_box[x, bouding_box_indices[1][1], slice] = color
        for y in range(bouding_box_indices[1][0], bouding_box_indices[1][1] - 1):
            volume_box[bouding_box_indices[0][0], y, slice] = color
            volume_box[bouding_box_indices[0][1], y, slice] = color
    return volume_box


def load_model_and_test_data(train: NDict, paths: NDict, infer: NDict):
    lgr = logging.getLogger('Fuse')

    checkpoint_file = os.path.join(paths["model_dir"], infer["checkpoint"])
    lgr.info(f'checkpoint_file={checkpoint_file}', {'color': 'magenta'})

    # load model
    lgr.info('Model:', {'attrs': 'bold'})
    model, pl_trainer, num_classes, gt_label, class_names = create_model(train, paths)
    lgr.info('Model: Done', {'attrs': 'bold'})

    ## Data
    folds = load_pickle(os.path.join(paths["data_misc_dir"], paths["data_split_filename"]))  # assume exists and created in train func

    infer_sample_ids = []
    for fold in infer["infer_folds"]:
        infer_sample_ids += folds[fold]
    input_source_gt = read_clinical_data_file(paths["clinical_data_file"], infer['target'], infer.get('columns_to_add'))

    test_dataset = UKBB.dataset(data_dir=paths["data_dir"], target=infer['target'], series_config=train['series_config'],
                                input_source_gt=input_source_gt, cache_dir=paths["cache_dir"], num_workers=infer['num_workers'],
                                sample_ids=infer_sample_ids, train=False)
    ## Create dataloader
    infer_dataloader = DataLoader(dataset=test_dataset,
                                  shuffle=False, drop_last=False,
                                  collate_fn=CollateDefault(),
                                  num_workers=infer["num_workers"])

    pl_module = LightningModuleDefault.load_from_checkpoint(checkpoint_file, model_dir=paths["model_dir"], model=model, map_location="cpu",
                                                            strict=True)

    return pl_module, pl_trainer, infer_dataloader


def show_attention_on_image(img: np.ndarray,
                            mask: np.ndarray,
                            colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    reference for fusing heat map and original image : https://github.com/jacobgil/pytorch-grad-cam/blob/61e9babae8600351b02b6e90864e4807f44f2d4a/pytorch_grad_cam/utils/image.py#L25
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmaps = [np.float32(cv2.applyColorMap(np.uint8(255 * mask[i]), colormap)) / 255 for i in range(mask.shape[0])]
    images = [cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB) for i in range(img.shape[0])]
    RGB_DTYPE = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    cams = []
    for i in range(len(images)):
        if np.max(images[i]) > 1:
            images[i] *= (1.0 / images[i].max())

        cam = heatmaps[i] + images[i]
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        cam = cam.view(RGB_DTYPE)
        cams.append(cam)
    nifti = nib.Nifti1Image(np.concatenate(cams, axis=2), np.eye(4))
    return nifti


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


######################################
# Analyze Template
######################################
def run_eval(paths: NDict, infer: NDict):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Eval', {'attrs': ['bold', 'underline']})

    # metrics
    metrics = OrderedDict([
        ('op', MetricApplyThresholds(pred='model.output.head_0')),  # will apply argmax
        ('auc', MetricAUCROC(pred='model.output.head_0', target='data.gt.classification')),
        ('accuracy', MetricAccuracy(pred='results:metrics.op.cls_pred', target='data.gt.classification')),
    ])

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(ids=None,
                             data=os.path.join(paths["inference_dir"], infer["infer_filename"]),
                             metrics=metrics,
                             output_dir=paths["eval_dir"])

    return results


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = NDict(OmegaConf.to_object(cfg))
    print(cfg)
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    choose_and_enable_multiple_gpus(cfg["train.trainer.devices"], force_gpus=force_gpus)

    # train
    if 'train' in cfg["run.running_modes"]:
        run_train(cfg["paths"], cfg["train"])
    else:
        assert "Expecting train mode to be set."

    # infer (infer set)
    if 'infer' in cfg["run.running_modes"]:
        run_infer(cfg["train"], cfg["paths"], cfg["infer"])
    #
    # evaluate (infer set)
    if 'eval' in cfg["run.running_modes"]:
        run_eval(cfg["paths"], cfg["infer"])

    # explain (infer set)
    if 'explain' in cfg["run.running_modes"]:
        run_explain(cfg["train"], cfg["paths"], cfg["infer"])


def read_clinical_data_file(filename, target, columns_to_add, return_var_namespace=False):
    df = pd.read_csv(filename)
    if (target not in df.columns) or return_var_namespace:
        var_namespace = cohort_and_label_def.get_clinical_vars_namespace(df, columns_to_add)
        assert target in var_namespace
        df[target] = var_namespace[target]
    df[target] = df[target].astype(int)
    if return_var_namespace:
        return df, var_namespace
    return df


if __name__ == "__main__":
    sys.argv.append('hydra.run.dir=working_dir')
    main()
