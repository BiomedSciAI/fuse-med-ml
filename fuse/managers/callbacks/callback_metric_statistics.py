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
import traceback
from typing import Dict, List, Optional

import pandas as pd

from fuse.managers.callbacks.callback_base import FuseCallback
from fuse.utils.utils_file import FuseUtilsFile as file
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseMetricStatisticsCallback(FuseCallback):
    """
        Responsible of writing the metric results into a CSV file under output_path
        The columns are: mode, epoch, metric_name, metric_value
    """

    def __init__(self, output_path: str, metrics: Optional[List[str]] = None) -> None:
        """
        :param output_path: output csv file path
        :param metrics: list of values out of the epoch_result dictionary to save.
                        Set to, will output all the epoch results to a csv file.
        """
        super().__init__()
        self.output_path = output_path
        self.metrics = metrics

        # prepare output_path dir (if not already exists)
        file.create_dir(os.path.dirname(self.output_path))

        # create file on first attempt to write values
        self.first = True

        pass

    def on_epoch_end(self, mode: str, epoch: int, epoch_results: Dict = None) -> None:
        """
        Adds a line to a CSV with the metrics results of self.metrics.
        The CSV is saved under self.output_path

        :param mode: either 'train', 'validation' or 'infer'.
        :param epoch: epoch number
        :param epoch_results: contains the data to be saved to CSV.
            e.g.: {'losses': {'loss1': mean_loss1,
                               'loss2': mean_loss2,
                                'total_loss': mean (loss1 + loss2)}
                    'metrics': {'metric1': epoch_metric1,
                                'metric2': epoch_metric2}}
        """
        if epoch_results is None:
            return

        lgr = logging.getLogger('Fuse')
        flat_results = FuseUtilsHierarchicalDict.flatten(epoch_results)
        # if this is step 0 - pre-training results should be logged in addition to the metrics csv
        if epoch == 0:
            lgr.info(f'Stats for Pre-Training:')
            for evaluator_name, evaluator_value in sorted(flat_results.items()):
                lgr.info(f'{evaluator_name:20} = {str(evaluator_value)}')

        # create a file
        if self.first:
            header_df = pd.DataFrame(columns=['mode', 'epoch'] + sorted(flat_results.keys()))
            header_df.to_csv(self.output_path, header=True, index=False)
            self.first = False

        metrics_df = pd.DataFrame.from_dict(flat_results, orient='index').T
        metrics_df['mode'] = mode
        metrics_df['epoch'] = epoch

        try:
            if self.metrics is None:
                metrics_df[['mode', 'epoch'] + sorted(flat_results.keys())].to_csv(self.output_path, mode='a', header=False, index=False)
            else:
                metrics_df[['mode', 'epoch'] + self.metrics].to_csv(self.output_path, mode='a', header=False, index=False)
        except Exception as e:
            track = traceback.format_exc()
            lgr.error(track)
            lgr.error("Cannot write epoch stats file (maybe it is open in another program?). Will try again next epoch.")

        pass
