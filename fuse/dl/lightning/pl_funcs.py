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
import traceback
from typing import Any, Dict, List, OrderedDict, Sequence, Union
from statistics import mean
from fuse.data.utils.sample import get_sample_id_key
from fuse.utils.data.collate import uncollate
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from .pl_epoch_summary import ModelEpochSummary

from fuse.utils import NDict
from fuse.dl.losses.loss_base import LossBase
from fuse.eval import MetricBase
from fuse.eval.metrics.utils import PerSampleData


def model_checkpoint_callbacks(model_dir: str, best_epoch_source: Union[Dict, List[Dict]]) -> List[pl.Callback]:
    """
    Create list of pl.callbacks that saves checkpoints using (pl.callbacks.ModelCheckpoint) and print per epoch summary (fuse.dl.lightning.pl_epoch_summary.ModelEpochSummary).
    :param model_dir: path to save checkpoints and summary
    :param best_epoch_source: either a dict with arguments to pass to ModelCheckpoint or list dicts to for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
    """
    callbacks = []
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
    return callbacks


def convert_predictions_to_dataframe(predictions: List[NDict]) -> pd.DataFrame:
    """list of batch_dict to a dataframe"""
    assert len(predictions) > 0

    values = {}
    predictions_per_sample = []
    for elem in predictions:
        predictions_per_sample += uncollate(elem)
    keys = predictions_per_sample[0].keypaths()
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
        pl.log(f"{mode}{sep}losses.{key}", loss, on_epoch=True, sync_dist=True)


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
            pl.log(f"{mode}{sep}{key}", epoch_results[key], on_epoch=True, sync_dist=True)
