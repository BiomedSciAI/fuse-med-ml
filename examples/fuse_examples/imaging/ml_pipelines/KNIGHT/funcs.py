import torchvision
from torchvision import transforms
from fuse.utils.utils_logger import fuse_logger_start
import logging
from fuse.data.utils.samplers import BatchSamplerDefault
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torchvision.models as models
from typing import OrderedDict
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve, MetricConfusion
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds

import torch.optim as optim
from fuse.utils import gpu as FuseUtilsGPU
import os
import torch.nn.functional as F
from fuse.eval.evaluator import EvaluatorDefault 
import copy

from examples.fuse_examples.imaging.classification.knight.make_predictions_file import make_predictions_file
from examples.fuse_examples.imaging.classification.knight.make_targets_file import make_targets_file
from examples.fuse_examples.imaging.classification.knight.eval.eval import eval
from fuse.data.utils.collates import CollateDefault
from fuse.dl.lightning.pl_module import LightningModuleDefault
import pytorch_lightning as pl
from fuse.dl.losses.loss_default import LossDefault
import pandas as pd
import numpy as np
from fuseimg.datasets.knight import KNIGHT
from examples.fuse_examples.imaging.classification.knight.baseline.fuse_baseline import make_model

def run_train(dataset, sample_ids, cv_index, test=False, params=None, \
        rep_index=0, rand_gen=None):
    assert(test == False)
    model_dir = os.path.join(params["paths"]["model_dir"], 'rep_' + str(rep_index), str(cv_index))
    cache_dir = os.path.join(params["paths"]["cache_dir"], 'rep_' + str(rep_index), str(cv_index))

    # start logger
    fuse_logger_start(output_path=model_dir, console_verbose_level=logging.INFO)
    print("Done")

    # obtain train/val dataset subset:
    ## Create subset data sources

    split = {}
    split["train"] = [params["dataset"]["split"]["train"][i] for i in sample_ids[0]]
    split["val"] = [params["dataset"]["split"]["train"][i] for i in sample_ids[1]]

    train_dataset, validation_dataset = KNIGHT.dataset(data_path=params["paths"]["data_dir"],
                                                        cache_dir=cache_dir,
                                                        split=split,
                                                        sample_ids=None,
                                                        test=False,
                                                        reset_cache=True,
                                                        resize_to=params["dataset"]["resize_to"])

    num_classes = params["common"]["num_classes"]
    ## Create sampler
    print("- Create sampler:")
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name=params["common"]["target_name"],
        num_balanced_classes=num_classes,
        batch_size=params["batch_size"],
        balanced_class_weights=[1.0 / num_classes] * num_classes if params["common"]["task_num"] == 2 else None,
    )

    print("- Create sampler: Done")

    ## Create dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=8,
        generator=rand_gen,
    )
    print("Train Data: Done", {"attrs": "bold"})

    ## Create dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=params["batch_size"],
        num_workers=8,
        collate_fn=CollateDefault(),
        generator=rand_gen,
    )
    print("Validation Data: Done", {"attrs": "bold"})


    ## Model definition
    ##############################################################################

    model = make_model(params["common"]["use_data"], num_classes, params["imaging_dropout"], params["fused_dropout"])


    # Loss definition:
    ##############################################################################
    losses = {
        "cls_loss": LossDefault(pred="model.logits.head_0", target=params["common"]["target_name"], callable=F.cross_entropy, weight=1.0)
    }


    # Metrics definition:
    ##############################################################################
    train_metrics = OrderedDict(
        [
            ("op", MetricApplyThresholds(pred="model.output.head_0")),  # will apply argmax
            ("auc", MetricAUCROC(pred="model.output.head_0", target=params["common"]["target_name"])),
            ("accuracy", MetricAccuracy(pred="results:metrics.op.cls_pred", target=params["common"]["target_name"])),
            (
                "sensitivity",
                MetricConfusion(pred="results:metrics.op.cls_pred", target=params["common"]["target_name"], metrics=("sensitivity",)),
            ),
        ]
    )
    val_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    best_epoch_source = dict(
        monitor=params["common"]["target_metric"],  # can be any key from losses or metrics dictionaries
        mode="max",  # can be either min/max
    )

    # Optimizer definition:
    ##############################################################################
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=0.001)

    # Scheduler definition:
    ##############################################################################
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    ## Training
    ##############################################################################

    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=model_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )
    # create lightining trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=model_dir,
        max_epochs=params["num_epochs"],
        accelerator="gpu",
        devices=params["common"]["num_gpus_per_split"],
        strategy=None,
        auto_select_gpus=False,
        num_sanity_val_steps=-1,
    )

    # train
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader, ckpt_path=None)


def run_infer(dataset, sample_ids, cv_index, test=False, params=None, \
              rep_index=0, rand_gen=None):

    model_dir = os.path.join(params["paths"]["model_dir"], 'rep_' + str(rep_index), str(cv_index))
    cache_dir = os.path.join(params["paths"]["cache_dir"], 'infer_cache', 'rep_' + str(rep_index), str(cv_index))
    if test:
        infer_dir = os.path.join(params["paths"]["test_dir"], 'rep_' + str(rep_index), str(cv_index))
    else:
        infer_dir = os.path.join(params["paths"]["inference_dir"], 'rep_' + str(rep_index), str(cv_index))

    if not os.path.isdir(infer_dir):
        os.makedirs(infer_dir)

    checkpoint = os.path.join(model_dir, "best_epoch.ckpt")
    data_path = params['paths']['data_dir']
    predictions_filename = os.path.join(infer_dir, 'predictions.csv')
    targets_filename = os.path.join(infer_dir, 'targets.csv')

    predictions_key_name = "model.output.head_0"
    task_num = params['common']['task_num']

    split = {}
    if sample_ids is not None:
        split["train"] = [params["dataset"]["split"]["train"][i] for i in sample_ids[0]]
        split["val"] = [params["dataset"]["split"]["train"][i] for i in sample_ids[1]]
    else:
        # test mode
        split["train"] = params["dataset"]["split"]["train"] # we will not use this
        split["val"] = params["dataset"]["split"]["val"]

    model = make_model(params["common"]["use_data"], params["common"]["num_classes"], params["train"]["imaging_dropout"], params["train"]["fused_dropout"])

    make_predictions_file(model_dir=model_dir, model=model, checkpoint=checkpoint, data_path=data_path, cache_path=cache_dir, \
        split=split, output_filename=predictions_filename, predictions_key_name=predictions_key_name, \
        task_num=task_num, auto_select_gpus=False, reset_cache=True)
    make_targets_file(data_path=data_path, split=split, output_filename=targets_filename)

def run_eval(dataset, sample_ids, cv_index, test=False, params=None, \
             rep_index=0, rand_gen=None, pred_key='model.output.classification', \
             label_key='data.label'):

    if test:
        infer_dir = os.path.join(params["paths"]["test_dir"], 'rep_' + str(rep_index), str(cv_index))
    else:
        infer_dir = os.path.join(params["paths"]["inference_dir"], 'rep_' + str(rep_index), str(cv_index))
    targets_filename = os.path.join(infer_dir, 'targets.csv')
    predictions_filename = os.path.join(infer_dir, 'predictions.csv')
    output_dir = infer_dir
    if test:
        if not cv_index=='ensemble':
            # individual fold test evaluation mode
            # combine preds and targets into a single file for the ensemble
            # and save in fuse inference-like format required for the ensemble metric pre_collect_process_func
            unified_filepath = os.path.join(infer_dir, params['test_infer_filename'])
            preds = pd.read_csv(predictions_filename)
            targets = pd.read_csv(targets_filename)
            df = pd.DataFrame(columns = ['id','model.output.classification','data.label'])
            df["id"] = preds['Case_id']
            df['model.output.classification']=list(np.stack((preds['NoAT-score'], preds['CanAT-score']),1))
            df['data.label'] = targets['Task1-target']        
            df.to_pickle(unified_filepath)
        else:
            # ensemble evaluation mode
            # save results in csv format expected by KNIGHT's evaluation function
            ensemble_res = pd.read_pickle(os.path.join(infer_dir,'ensemble_results.gz'))

            pred_df = pd.DataFrame(columns = ['Case_id','NoAT-score','CanAT-score'])
            pred_df["Case_id"] = ensemble_res['id']
            pred_df["NoAT-score"] = [item[0] for item in list(ensemble_res.preds)]
            pred_df["CanAT-score"] = [item[1] for item in list(ensemble_res.preds)]
            pred_df.to_csv(predictions_filename, index=False)

            target_df = pd.DataFrame(columns = ['case_id','Task1-target'])
            target_df["case_id"] = ensemble_res['id']
            target_df["Task1-target"] = ensemble_res['target']
            target_df.to_csv(targets_filename, index=False)

    eval(target_filename=targets_filename, task1_prediction_filename=predictions_filename, task2_prediction_filename="", output_dir=output_dir)


