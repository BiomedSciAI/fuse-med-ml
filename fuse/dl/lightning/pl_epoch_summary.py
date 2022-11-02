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

from copy import deepcopy
import os
from typing import Optional
from fuse.utils.misc.misc import get_pretty_dataframe

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback

from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch
import pandas as pd


class ModelEpochSummary(Callback):
    """
        Model Checkpointing Display
        ===================

        Automatically display (print to screen and log to a file) best vs current epoch metircs and losses.

        Example:
    Stats for epoch: 9 (best tpoch is 7 for source validation.metrics.accuracy!)
    ------------------------------------------------------------------------------------------
    |                             | Best Epoch (7)              | Current Epoch (9)           |
    ------------------------------------------------------------------------------------------
    | train.losses.cls_loss       | 0.1546                      | 0.1146                      |
    ------------------------------------------------------------------------------------------
    | train.losses.total_loss     | 0.1546                      | 0.1146                      |
    ------------------------------------------------------------------------------------------
    | train.metrics.accuracy      | 0.9545                      | 0.9655                      |
    ------------------------------------------------------------------------------------------
    | validation.losses.cls_loss  | 0.0720                      | 0.0920                      |
    ------------------------------------------------------------------------------------------
    | validation.losses.total_loss| 0.0720                      | 0.0920                      |
    ------------------------------------------------------------------------------------------
    | validation.metrics.accuracy | 0.9776                      | 0.9756                      |
    ------------------------------------------------------------------------------------------

    """

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        mode: str = "min",
    ):
        """
        :param dirpath: location of log file
        :param filename: specify a filename. If not set will use f"epoch_summary_{monitor}.txt"
        :param monitor: the metric name to track
        :param mode: either consider the "min" value to be best or the "max" value to be the best
        """
        super().__init__()
        self._monitor = monitor
        self._mode = mode
        self._dirpath = dirpath
        self._filename = filename if filename is not None else f"epoch_summary_{self._monitor.replace('/', '.')}.txt"
        self._best_epoch_metrics = None
        self._best_epoch_index = None

    @rank_zero_only
    def print_epoch_summary_table(self, epoch_metrics: dict, epoch_source_index: int) -> None:
        """
        Generate, print and log the epoch summary table.
        Decorator makes sure it runs only once in a DDP strategy.
        """

        def get_value_as_float_str(dict, key):
            val_as_str = "N/A"
            try:
                value = dict[key]
                val_as_str = "%.4f" % float(value)
            except:
                pass
            return val_as_str

        stats_table = pd.DataFrame(
            columns=["", f"Best Epoch ({self._best_epoch_index})", f"Current Epoch ({epoch_source_index})"]
        )
        idx = 0

        eval_keys = sorted(epoch_metrics.keys())
        for evaluator_name in eval_keys:
            current_str = get_value_as_float_str(epoch_metrics, evaluator_name)
            best_so_far_str = get_value_as_float_str(self._best_epoch_metrics, evaluator_name)

            stats_table.loc[idx] = [f"{evaluator_name}", best_so_far_str, current_str]
            idx += 1

        if self._best_epoch_index == epoch_source_index:
            epoch_title = (
                f"Stats for epoch: {epoch_source_index} (Currently the best epoch for source {self._monitor}!)"
            )
            print(epoch_title)
        else:
            epoch_title = f"Stats for epoch: {epoch_source_index} (Best epoch is {self._best_epoch_index} for source {self._monitor})"
            print(epoch_title)
        stats_as_str = get_pretty_dataframe(stats_table)
        print(stats_as_str)
        print(f"Model: {self._dirpath}")

        try:
            op = "w" if epoch_source_index == 0 else "a"  # we want to have all stats in one file per epoch
            with open(os.path.join(self._dirpath, self._filename), op) as sfile:
                sfile.write(epoch_title)
                sfile.write(stats_as_str)
        except Exception as error:
            print(f"Cannot write epoch summary to file {self._dirpath}/{self._filename}")
            print(error)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Print summary at the end of the validation stage."""
        monitor_candidates = monitor_candidates = deepcopy(trainer.callback_metrics)
        current_epoch_metrics = monitor_candidates
        monitor_op = {"min": torch.lt, "max": torch.gt}[self._mode]
        if self._best_epoch_metrics is None or monitor_op(
            current_epoch_metrics[self._monitor], self._best_epoch_metrics[self._monitor]
        ):
            self._best_epoch_metrics = current_epoch_metrics
            self._best_epoch_index = trainer.current_epoch
        self.print_epoch_summary_table(current_epoch_metrics, trainer.current_epoch)
