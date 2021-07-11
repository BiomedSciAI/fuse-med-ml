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
import time
from typing import Dict

import torch.nn as nn

from fuse.managers.callbacks.callback_base import FuseCallback
from fuse.managers.manager_state import FuseManagerState
from fuse.utils import utils_misc as misc


class FuseTimeStatisticsCallback(FuseCallback):
    """
        Counts time of procedures.
    """

    def __init__(self, num_epochs: int, load_expected_part=0.1) -> None:
        """

        :param num_epochs: total number of epochs (to compute remaining time)
        :param load_expected_part: expected fraction of the loading from the total handle_epoch time.
        """
        super().__init__()
        self.num_epochs = num_epochs
        self.load_expected_part = load_expected_part

        self.step_begin_time = None
        self.epoch_begin_time = None
        self.batch_begin_time = None
        self.virtual_batch_begin_time = None
        self.load_batch_aggregated_time = 0
        self.train_begin_time = None

        pass

    def on_step_begin(self, step: int) -> None:
        self.step_begin_time = time.time()
        pass

    def on_step_end(self, step: int, train_results: Dict = None, validation_results: Dict = None, learning_rate: float = None) -> None:
        """
        Counts the time of a step (train + validation + update scheduler), and computes the remaining time for the entire run.

        :param step: step number (out of self.num_epochs)
        :param train_results: ignored
        :param validation_results: ignored
        :param learning_rate: ignored
        """
        end_time = time.time()
        timedelta_seconds = end_time - self.step_begin_time
        running_time_left_seconds = timedelta_seconds * (self.num_epochs - step)
        running_time_left_str = misc.time_display(running_time_left_seconds)
        epoch_time_str = misc.time_display(timedelta_seconds)
        logging.getLogger('Fuse').info(f'Estimated time left: {running_time_left_str} (last epoch time: {epoch_time_str})', {'color': 'cyan'})
        pass

    def on_epoch_begin(self, mode: str, epoch: int) -> None:
        """
        reset epoch_begin_time and the load_batch_aggregated time
        :param mode: ignored
        :param epoch: ignored
        """
        self.epoch_begin_time = time.time()
        self.load_batch_aggregated_time = 0
        pass

    def on_epoch_end(self, mode: str, epoch: int, epoch_results: Dict = None) -> None:
        """
        Writes the time for epoch, and checks whether the loading of data took more than load_expected_part.
        If the loading fraction is greater than expected, notifies logger.

        :param mode: mode
        :param epoch: epoch number
        :param epoch_results: ignored
        """
        epoch_end = time.time()

        lgr = logging.getLogger('Fuse')
        lgr.debug(f"Time for {mode} epoch {epoch}: {misc.get_time_delta(self.epoch_begin_time)}")

        # debug info about the batch iterator loading time
        load_time_str = misc.time_display(self.load_batch_aggregated_time)
        lgr.debug(f'Mode: {mode}, epoch {epoch}: Total time for loading the data is {load_time_str}')

        # check if the loading part of the data exceeded load_expected_part of epoch time
        epoch_time = epoch_end - self.epoch_begin_time
        max_time_for_load = epoch_time * self.load_expected_part
        if self.load_batch_aggregated_time > max_time_for_load:
            lgr.info(f'Mode: {mode}, epoch {epoch}: '
                     f'Total time for loading data ({load_time_str}) is greater than expected (maximum {misc.time_display(max_time_for_load)})',
                     {'color': 'blue'})
        pass

    def on_virtual_batch_begin(self, mode: str, virtual_batch: int) -> None:
        self.virtual_batch_begin_time = time.time()
        pass

    def on_virtual_batch_end(self, mode: str, virtual_batch: int, virtual_batch_results: Dict = None) -> None:
        logging.getLogger('Fuse').debug(f"Time for {mode} virtual batch {virtual_batch}: {misc.get_time_delta(self.virtual_batch_begin_time)}")
        pass

    def on_batch_begin(self, mode: str, batch: int) -> None:
        self.batch_begin_time = time.time()
        pass

    def on_data_fetch_end(self, mode: str, batch: int, batch_dict: Dict = None) -> None:
        # add the delta time to load to the aggregated sum
        self.load_batch_aggregated_time += (time.time() - self.batch_begin_time)
        pass

    def on_batch_end(self, mode: str, batch: int, batch_dict: Dict = None) -> None:
        logging.getLogger('Fuse').debug(f"Time for {mode} batch {batch}: {misc.get_time_delta(self.batch_begin_time)}")
        pass

    def on_train_begin(self, state: FuseManagerState) -> None:
        # update number of epochs from the manager's state:
        self.num_epochs = state.num_epochs
        self.train_begin_time = time.time()
        pass

    def on_train_end(self) -> None:
        logging.getLogger('Fuse').debug(f"Time for train: {misc.get_time_delta(self.train_begin_time)}")
        pass
