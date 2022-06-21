"""
Model Checkpointing Display
===================

Automatically display (print to screen and save to file) best vs current epoch metircs.

"""
from copy import deepcopy
import os
from typing import Optional
from fuse.utils.misc.misc import get_pretty_dataframe

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
import torch
import pandas as pd


class ModelEpochSummary(Callback):
    
    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        mode: str = "min",
    ):
        super().__init__()
        self._monitor = monitor
        self._mode = mode
        self._dirpath = dirpath
        self._filename = filename if filename is not None else f"epoch_summary_{self._monitor}.txt"
        self._best_epoch_metrics = None
        self._best_epoch_index = None

    
    def print_epoch_summary_table(self, epoch_metircs: dict, epoch_source_index: int) -> None:
        def get_value_as_float_str(dict, key):
            val_as_str = 'N/A'
            try:
                value = dict[key]
                val_as_str = '%.4f' % float(value)
            except:
                pass
            return val_as_str

        stats_table = pd.DataFrame(columns=['', f'Best Epoch ({epoch_source_index})', f'Current Epoch ({self._best_epoch_index})'])
        idx = 0

        eval_keys = sorted(epoch_metircs.keys())
        for evaluator_name in eval_keys:
            current_str = get_value_as_float_str(epoch_metircs, evaluator_name)
            best_so_far_str = get_value_as_float_str(self._best_epoch_metrics, evaluator_name)

            stats_table.loc[idx] = [f'{evaluator_name}', best_so_far_str, current_str]
            idx += 1

        if self._best_epoch_index == epoch_source_index:
            epoch_title = f'Stats for epoch: {epoch_source_index} (Currently the best epoch for source {self._monitor}!)'
            print(f"{epoch_title}\n")
        else:
            epoch_title = f'Stats for epoch: {epoch_source_index} (Best epoch is {self._best_epoch_index} for source {self._monitor})'
            print(epoch_title)
        stats_as_str = get_pretty_dataframe(stats_table)
        print(stats_as_str)
        print(f"Model: {self._dirpath}")

        try:
            op = 'w' if epoch_source_index == 0 else 'a'  # we want to have all stats in one file per epoch
            with open(os.path.join(self._dirpath, self._filename), op) as sfile:
                sfile.write(epoch_title)
                sfile.write(stats_as_str)
        except Exception as error:
            print(f"Cannot write epoch summary to file {self._dirpath}/{self._filename}")
            print(error)
   

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the validation stage."""
        monitor_candidates = monitor_candidates = deepcopy(trainer.callback_metrics)  
        current_epoch_metrics = monitor_candidates
        monitor_op = {"min": torch.lt, "max": torch.gt}[self._mode]
        if self._best_epoch_metrics is None or monitor_op(current_epoch_metrics[self._monitor], self._best_epoch_metrics[self._monitor]):
            self._best_epoch_metrics = current_epoch_metrics
            self._best_epoch_index = trainer.current_epoch
        self.print_epoch_summary_table(current_epoch_metrics, trainer.current_epoch)
