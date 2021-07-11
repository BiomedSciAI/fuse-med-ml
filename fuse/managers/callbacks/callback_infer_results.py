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
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from fuse.managers.callbacks.callback_base import FuseCallback
from fuse.utils.utils_file import FuseUtilsFile as file
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseInferResultsCallback(FuseCallback):
    """
        Responsible of writing the data of inference results into a CSV file.
        Collects the output data (corresponding to the output_columns) at the end of handle_batch into an aggregated dict,
            and writes it at the end of handle_epoch.
        The method self.get_infer_results() may be used to get the aggregated dict.
    """

    def __init__(self, output_file: Optional[str] = None, output_columns: Optional[List[str]] = None) -> None:
        super().__init__()
        self.output_columns = output_columns
        self.output_file = output_file

        # prepare output_path (if not already exists)
        if self.output_file is not None:
            file.create_dir(os.path.dirname(self.output_file))

        self.reset()
        pass

    def reset(self):
        self.aggregated_dict = {'descriptor': [], 'output': {}}
        self.infer_results_df = pd.DataFrame()

    def on_epoch_begin(self, mode: str, epoch: int) -> None:
        # make sure the aggregated dict is empty
        self.reset()
        pass

    def get_infer_results(self) -> pd.DataFrame:
        """
        Returns the aggregated results dict.
        :return: contains the key descriptor and a key for each metric/loss in output_columns.
        """

        return self.infer_results_df

    def on_epoch_end(self, mode: str, epoch: int, epoch_results: Dict = None) -> None:
        """
        When mode = 'infer' create a pickled DataFrame with inference results of self.output_columns.
        File is saved under self.output_path

        :param mode: write to file only on 'infer' mode.
        :param epoch: epoch number (ignored)
        :param epoch_results: not actually used
        """
        if mode != 'infer':
            return

        # prepare dataframe from the results
        infer_results_df = pd.DataFrame()
        infer_results_df['descriptor'] = self.aggregated_dict['descriptor']

        for output in FuseUtilsHierarchicalDict.get_all_keys(self.aggregated_dict['output']):
            infer_results_df[output] = list(
                FuseUtilsHierarchicalDict.get(self.aggregated_dict['output'], output))  # note- wrapping with list for pandas compatibility

        if self.output_file is not None:
            infer_results_df.to_pickle(self.output_file, compression='gzip')
            logging.getLogger('Fuse').info(f"Save inference results into {self.output_file}")

        self.reset()
        self.infer_results_df = infer_results_df
        return

    def on_batch_end(self, mode: str, batch: int, batch_dict: Dict = None) -> None:
        """
        On batch end - save the descriptor data into a class member.

        :param mode: accept only infer mode
        :param batch: batch number
        :param batch_dict: the batch input data
        """
        if mode != 'infer':
            return

        # for infer we need the descriptor and the output predictions
        descriptors = batch_dict['data'].get('descriptor', None)
        if isinstance(descriptors, Tensor):
            descriptors = list(descriptors.detach().cpu().numpy())
        self.aggregated_dict['descriptor'].extend(descriptors)

        if self.output_columns is not None and len(self.output_columns) > 0:
            output_cols = self.output_columns
        else:
            output_cols = FuseUtilsHierarchicalDict.get_all_keys(batch_dict)

        for output_col in output_cols:
            if output_col not in FuseUtilsHierarchicalDict.get_all_keys(self.aggregated_dict['output']):
                FuseUtilsHierarchicalDict.set(self.aggregated_dict['output'], output_col, [])
            output = FuseUtilsHierarchicalDict.get(batch_dict, output_col)
            if isinstance(output, torch.Tensor):
                # no need to save tensors
                output = output.cpu().numpy()
            FuseUtilsHierarchicalDict.get(self.aggregated_dict['output'], output_col).extend(output)

        pass
