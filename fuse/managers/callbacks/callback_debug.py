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

import logging
from datetime import datetime
from typing import Dict

import torch.nn as nn

from fuse.managers.callbacks.callback_base import FuseCallback
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
from fuse.utils.utils_misc import FuseUtilsMisc


class FuseCallbackDebug(FuseCallback):
    """
    Callback used to log the information about each stage: begin/end time and fuse main structure: batch_dict, virtual_batch_results and
    epoch_results
    """

    def __init__(self):
        super().__init__()
        self.time_step_begin = None
        self.time_epoch_begin = None
        self.time_virtual_batch_begin = None
        self.time_batch_begin = None

        self.first_step = True
        self.first_epoch = {}
        self.first_virtual_batch = {}
        self.first_batch = {}

    def on_step_begin(self, step: int) -> None:
        # log info on first step only
        if self.first_step:
            self.time_step_begin = datetime.now()
            current_time = self.time_step_begin.strftime("%H:%M:%S")
            logging.getLogger('Fuse').info(f'\nStep {step} - {current_time} - BEGIN', {'color': 'green', 'attrs': ['bold', 'underline']})

    def on_step_end(self, step: int, train_results: Dict = None, validation_results: Dict = None, learning_rate: float = None) -> None:
        # log info on first step only: learning rate, train results and validation results
        if self.first_step:
            lgr = logging.getLogger('Fuse')

            time_step_end = datetime.now()
            current_time = time_step_end.strftime("%H:%M:%S")
            total_time = time_step_end - self.time_step_begin
            lgr.info(f'Step {step} - {current_time} (total time {total_time}) - END', {'color': 'green', 'attrs': ['bold', 'underline']})

            lgr.info(f'Step Results:', {'color': 'green', 'attrs': 'bold'})
            lgr.info(f'Learning Rate: {learning_rate}', {'color': 'green'})
            lgr.info(f'Train Results:', {'color': 'green', 'attrs': 'bold'})
            lgr.info(f'{FuseUtilsHierarchicalDict.to_string(train_results)}', {'color': 'green'})
            lgr.info(f'Validation Results:', {'color': 'green', 'attrs': 'bold'})
            lgr.info(f'{FuseUtilsHierarchicalDict.to_string(validation_results)}', {'color': 'green'})

            self.time_step_begin = None
            self.first_step = False

    def on_epoch_begin(self, mode: str, epoch: int) -> None:
        # log info on first train/validation epoch only
        if self.first_epoch.get(mode, True):
            self.time_epoch_begin = datetime.now()
            current_time = self.time_epoch_begin.strftime("%H:%M:%S")
            logging.getLogger('Fuse').info(f'\n{mode.capitalize()} epoch {epoch} - {current_time} - BEGIN', {'color': 'green', 'attrs': ['bold', 'underline']})

    def on_epoch_end(self, mode: str, epoch: int, epoch_results: Dict = None) -> None:
        # log info on first train/validation epoch only: epoch results
        if self.first_epoch.get(mode, True):
            lgr = logging.getLogger('Fuse')

            time_epoch_end = datetime.now()
            current_time = time_epoch_end.strftime("%H:%M:%S")
            total_time = time_epoch_end - self.time_epoch_begin
            lgr.info(f'{mode.capitalize()}  epoch {epoch} - {current_time} (total time {total_time}) - END', {'color': 'green', 'attrs': ['bold', 'underline']})

            lgr.info(f'Epoch Results:', {'color': 'green', 'attrs': 'bold'})
            lgr.info(f'{FuseUtilsHierarchicalDict.to_string(epoch_results)}', {'color': 'green'})

            self.time_epoch_begin = None
            self.first_epoch[mode] = False

    def on_virtual_batch_begin(self, mode: str, virtual_batch: int) -> None:
        # log info on first train/validation virtual batch only
        if self.first_virtual_batch.get(mode, True):
            self.time_virtual_batch_begin = datetime.now()
            current_time = self.time_virtual_batch_begin.strftime("%H:%M:%S")
            logging.getLogger('Fuse').info(f'\n{mode.capitalize()} virtual batch {virtual_batch} - {current_time} - BEGIN', {'color': 'green', 'attrs': ['bold', 'underline']})

    def on_virtual_batch_end(self, mode: str, virtual_batch: int, virtual_batch_results: Dict = None) -> None:
        # log info on first train/validation virtual batch only: virtual batch results
        if self.first_virtual_batch.get(mode, True):
            lgr = logging.getLogger('Fuse')

            time_virtual_batch_end = datetime.now()
            current_time = time_virtual_batch_end.strftime("%H:%M:%S")
            total_time = time_virtual_batch_end - self.time_virtual_batch_begin
            lgr.info(f'{mode.capitalize()} virtual batch {virtual_batch} - {current_time} (total time {total_time}) - END', {'color': 'green', 'attrs': ['bold', 'underline']})

            lgr.info(f'Virtual Batch Results:', {'color': 'green', 'attrs': 'bold'})
            lgr.info(f'{FuseUtilsHierarchicalDict.to_string(virtual_batch_results)}', {'color': 'green'})

            self.time_virtual_batch_begin = None
            self.first_virtual_batch[mode] = False

    def on_batch_begin(self, mode: str, batch: int) -> None:
        # log info on first train/validation batch only
        if self.first_batch.get(mode, True):
            self.time_batch_begin = datetime.now()
            current_time = self.time_batch_begin.strftime("%H:%M:%S")
            logging.getLogger('Fuse').info(f'\n{mode.capitalize()} batch {batch} - {current_time} - BEGIN', {'color': 'green', 'attrs': ['bold', 'underline']})

    def on_data_fetch_end(self, mode: str, batch: int, batch_dict: Dict = None) -> None:
        # log info on first train/validation batch only: batch_dict
        if self.first_batch.get(mode, True):
            lgr = logging.getLogger('Fuse')

            time_batch_end = datetime.now()
            current_time = time_batch_end.strftime("%H:%M:%S")
            total_time = time_batch_end - self.time_batch_begin
            lgr.info(f'{mode.capitalize()} data fetch {batch} - {current_time} (total time {total_time}) - END',
                     {'color': 'green', 'attrs': ['bold', 'underline']})

            lgr.info(f'Batch Dict:', {'color': 'green', 'attrs': 'bold'})
            lgr.info(f'{FuseUtilsMisc.batch_dict_to_string(batch_dict)}', {'color': 'green'})

    def on_batch_end(self, mode: str, batch: int, batch_dict: Dict = None) -> None:
        # log info on first train/validation batch only: batch_dict
        if self.first_batch.get(mode, True):
            lgr = logging.getLogger('Fuse')

            time_batch_end = datetime.now()
            current_time = time_batch_end.strftime("%H:%M:%S")
            total_time = time_batch_end - self.time_batch_begin
            lgr.info(f'{mode.capitalize()} batch {batch} - {current_time} (total time {total_time}) - END', {'color': 'green', 'attrs': ['bold', 'underline']})

            lgr.info(f'Batch Dict:', {'color': 'green', 'attrs': 'bold'})
            lgr.info(f'{FuseUtilsMisc.batch_dict_to_string(batch_dict)}', {'color': 'green'})

            self.time_batch_begin = None
            self.first_batch[mode] = False

    def on_train_begin(self, net: nn.Module) -> None:
        lgr = logging.getLogger('Fuse')
        lgr.info(f'\nTrain - BEGIN', {'color': 'green', 'attrs': ['bold', 'underline']})

    def on_train_end(self) -> None:
        lgr = logging.getLogger('Fuse')
        lgr.info(f'\nTrain - DONE', {'color': 'green', 'attrs': ['bold', 'underline']})
