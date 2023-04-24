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

======================================

Collection of useful functions to implement FuseMedML pytorch lightning based module and train loop
"""
import os
import traceback
from typing import Any, Dict, List, OrderedDict, Sequence, Union, Mapping, TypeVar
from statistics import mean
from fuse.data.utils.sample import get_sample_id_key
from fuse.utils.data.collate import uncollate
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from .pl_epoch_summary import ModelEpochSummary

from fuse.utils import NDict
from fuse.dl.losses.loss_base import LossBase
from fuse.eval import MetricBase
from fuse.eval.metrics.utils import PerSampleData

# for clearml
from clearml import Task

TaskInstance = TypeVar("TaskInstance", bound="Task")


def start_clearml_logger(
    project_name: Union[str, None],
    task_name: Union[str, None],
    tags: Union[Sequence[str], None] = None,
    reuse_last_task_id: Union[bool, str] = True,
    continue_last_task: Union[bool, str, int] = False,
    output_uri: Union[str, bool, None] = None,
    auto_connect_arg_parser: Union[bool, Mapping[str, bool]] = True,
    auto_connect_frameworks: Union[bool, Mapping[str, bool]] = True,
    auto_resource_monitoring: bool = True,
    auto_connect_streams: Union[bool, Mapping[str, bool]] = True,
    deferred_init: bool = False,
) -> TaskInstance:
    """
    Just a fuse function to quickly start the clearml logger. It sets up patches to pytorch lightning logging hooks so it doesn't need to be passed to any lightning logger.
    This function also checks if the NODE_RANK and LOCAL_RANK env variables have been set. In which case clearml will only be initialized on global rank 0.
    For information on all the arguments please see: https://clear.ml/docs/latest/docs/references/sdk/task/ or https://github.com/allegroai/clearml/blob/master/clearml/task.py

    General Clearml instructions:
    Unless using offline mode, to use clearml, you must first make an account on their website https://app.clear.ml/login?redirect=%2Fsettings%2Fworkspace-configuration.
    Then, you must create a ~/clearml.conf file and specify server address as shown here https://clear.ml/docs/latest/docs/configs/clearml_conf/.
    Otherwise, offline mode instructions can be found here: https://clear.ml/docs/latest/docs/guides/set_offline/

    Example usage:
    from fuse.dl.lightning.pl_funcs import start_clearml_logger
    start_clearml_logger(project_name="my_project_name", task_name="test_01")
    """
    bool_start_logger = False
    task = None

    # check if we are in a distributed setting (if we are, must check that we are also on global rank 0)
    distributed = ("NODE_RANK" in os.environ) and ("LOCAL_RANK" in os.environ)
    if distributed:
        node_rank = int(os.environ["NODE_RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if (node_rank == 0) and (local_rank == 0):
            bool_start_logger = True
    else:
        # if not in a distributed setting, we can just start logger
        bool_start_logger = True

    if bool_start_logger:
        task = Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=tags,
            reuse_last_task_id=reuse_last_task_id,
            continue_last_task=continue_last_task,
            output_uri=output_uri,
            auto_connect_arg_parser=auto_connect_arg_parser,
            auto_connect_frameworks=auto_connect_frameworks,
            auto_resource_monitoring=auto_resource_monitoring,
            auto_connect_streams=auto_connect_streams,
            deferred_init=deferred_init,
        )
    return task


def model_default_callbacks(
    model_dir: str, best_epoch_source: Union[Dict, List[Dict], None], log_lr: bool = True
) -> List[pl.Callback]:
    """
    Create list of pl.callbacks that saves checkpoints using (pl.callbacks.ModelCheckpoint), print per epoch summary (fuse.dl.lightning.pl_epoch_summary.ModelEpochSummary) and log learning rate (lightning.pytorch.callbacks.LearningRateMonitor).
    :param model_dir: path to save checkpoints and summary
    :param best_epoch_source: either a dict with arguments to pass to ModelCheckpoint or list dicts to for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
                              If set to None, it only store the last checkpoint.
    """
    callbacks = []

    if best_epoch_source is None:
        model_checkpoint_display = ModelEpochSummary(dirpath=model_dir)
        callbacks.append(model_checkpoint_display)
    else:
        # checkpoints
        if not isinstance(best_epoch_source, list):
            best_epoch_source = [best_epoch_source]
        for checkpoint_to_monitor in best_epoch_source:
            if "dirpath" not in checkpoint_to_monitor:
                checkpoint_to_monitor["dirpath"] = model_dir
            if "filename" not in checkpoint_to_monitor:
                checkpoint_to_monitor["filename"] = "best_epoch"
                if len(best_epoch_source) > 1:
                    checkpoint_to_monitor["auto_insert_metric_name"] = True

            model_checkpoint = ModelCheckpoint(**checkpoint_to_monitor)
            model_checkpoint_display = ModelEpochSummary(
                dirpath=checkpoint_to_monitor["dirpath"],
                monitor=checkpoint_to_monitor.get("monitor", None),
                mode=checkpoint_to_monitor.get("mode", "min"),
            )
            callbacks.append(model_checkpoint)
            callbacks.append(model_checkpoint_display)

    # last epoch checkpoint
    callbacks.append(ModelCheckpoint(dirpath=model_dir, filename="last_epoch", save_last=True))

    if log_lr:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    return callbacks


def convert_predictions_to_dataframe(predictions: List[NDict]) -> pd.DataFrame:
    """list of batch_dict to a dataframe"""
    assert len(predictions) > 0

    values = {}
    predictions_per_sample = []
    for elem in predictions:
        predictions_per_sample += uncollate(elem)
    if isinstance(predictions_per_sample[0], NDict):
        keys = predictions_per_sample[0].keypaths()
    else:  # dict
        keys = predictions_per_sample[0].keys()

    for key in keys:
        values[key] = [elem[key] for elem in predictions_per_sample]

    df = pd.DataFrame(values)
    return df


def step_losses(losses: Dict[str, LossBase], batch_dict: NDict) -> torch.Tensor:
    """
    Compute losses per step (batch) in pl.LightningModule.<training/validation/test>_step()
    :param losses: dict of FuseMedML style losses
    :param batch_dict: FuseMedML batch_dict including data and model outputs
    :return: total_loss (sum all losses results). The values for tracking purpose will be stored in batch_dict['losses']
    """
    total_loss = None
    for loss_name, loss_function in losses.items():
        current_loss_result = loss_function(batch_dict)
        batch_dict["losses." + loss_name] = current_loss_result.data.item()
        # sum all losses for backward
        if total_loss is None:
            total_loss = current_loss_result
        else:
            total_loss += current_loss_result

    if total_loss is not None:
        batch_dict["losses.total_loss"] = total_loss.data.item()

    return total_loss


def step_metrics(metrics: OrderedDict[str, MetricBase], batch_dict: NDict) -> None:
    """
    Collect data to compute per epoch metrics
    :param metrics: dictionary of metrics
    :param batch_dict: FuseMedML batch_dict including data, losses and model outputs
    :return: None
    """
    for _, metric in metrics.items():
        # handle batch doesn't return a value, the actual value of the metric is per epoch
        metric.collect(batch_dict)


def step_extract_predictions(prediction_keys: Sequence[str], batch_dict: NDict) -> Dict[str, Any]:
    """
    Extract the specified predictions from batch_dict (torch Tensors will be detached, moved to cpu and coverted to numpy)
    :param prediction_keys: the keys to extract
    :param batch_dict: FuseMedML batch_dict including data and model outputs
    """
    outputs = {}
    sample_ids = batch_dict[get_sample_id_key()]
    if isinstance(sample_ids, torch.Tensor):
        sample_ids = list(sample_ids.detach().cpu().numpy())
    outputs["id"] = sample_ids
    for key in prediction_keys:
        output = batch_dict[key]
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        outputs[key] = output

    return outputs


def epoch_end_compute_and_log_losses(
    pl: pl.LightningModule, mode: str, batch_losses: Sequence[Dict], sep: str = "."
) -> None:
    """
    On epoch end average out the batch losses and log the averaged losses
    :param pl: LightningModule. Used for logging.
    :param mode: prefix to add to each loss name (when logging), typically validation/train/test
    :param batch_losses: list of batch_dict["losses"] as added by 'epoch_losses'
    :return: None
    """
    keys = batch_losses[0].keys()
    for key in keys:
        losses = []
        for elem in batch_losses:
            if isinstance(elem[key], torch.Tensor):
                losses.extend(elem[key].detach().cpu().tolist())
            else:
                losses.append(elem[key])
        loss = mean(losses)
        pl.log(f"{mode}{sep}losses.{key}", loss, on_epoch=True, sync_dist=True, rank_zero_only=True)


def epoch_end_compute_and_log_metrics(
    pl: pl.LightningModule, mode: str, metrics: OrderedDict[str, MetricBase], sep: str = "."
) -> None:
    """
    On epoch end compute and log per epoch metrics
    :param pl: LightningModule. Used for logging.
    :param mode: prefix to add to each metric name (when logging), typically validation/train/test
    :param metrics: Dict of FuseMedML style metrics
    :return: None
    """
    # compute metrics
    epoch_results = NDict()
    # compute metrics and keep the results
    for metric_name, metric in metrics.items():
        try:
            metric_result = metric.eval(epoch_results)
        except:
            track = traceback.format_exc()
            print(f"Metric {metric_name} process() func failed. Setting results to None")
            print(track)
            metric_result = None

        epoch_results[f"metrics.{metric_name}"] = metric_result
        metric.reset()

    # log metrics
    for key in epoch_results.keypaths():
        if epoch_results[key] is not None and not isinstance(epoch_results[key], (PerSampleData)):
            pl.log(f"{mode}{sep}{key}", epoch_results[key], on_epoch=True, sync_dist=True, rank_zero_only=True)
