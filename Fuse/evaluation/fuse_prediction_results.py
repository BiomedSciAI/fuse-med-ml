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

import ast
import logging
from typing import Callable

import pandas as pd

from Fuse.data.processor.processor_base import FuseProcessorBase
from Fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict

try:
    from class_definitions.label import Label
    from data_entities import prediction_results_entity

    CANNOT_LOAD_CLASSEVE = False
except Exception:
    CANNOT_LOAD_CLASSEVE = True


class FusePredictionResults(prediction_results_entity.UnifiedEntityPrediction):
    """
    DataEntity to contain the prediction results of a Fuse evaluation process.
    """

    def __init__(self, raw_data: pd.DataFrame = None, inference_filename: str = None,
                 output_column: str = None,
                 gt_processor: FuseProcessorBase = None, gt_label_name: str = 'tensor',
                 weights_function: Callable = None) -> None:
        """
        Creates a unified prediction result entity to be used by eve.
        :param inference_filename: CSV with inference results
        :param output_column: The column in CSV that contain the prediction results
        :param gt_processor: processor to retrieve the ground truth from
        :param gt_label_name: label of key in gt_processor output that contains the ground_truth data. label can be a hierarchical key.
        :param weights_function: weights function to compute the sample weights. Its input is a dataframe with the columns: key, score, ground_truth.
            The weights_function is expected to add at least two columns to the input dataframe: weight, stratum.
            see data_entities.compute_weight_stratum.compute_weight_by_distribution for example of computing weights.
        """
        if CANNOT_LOAD_CLASSEVE:
            raise Exception("ClassEve Cannot be loaded, please make sure it's on the PYTHONPATH")

        super().__init__()
        self.logger = logging.getLogger('Fuse')

        if raw_data is None and inference_filename is None:
            raise Exception("Cannot instansiate Prediction results: specify either 'raw_data' or 'inference_filename'")
        if raw_data is None:
            self.logger.info(f"Reading prediction file {inference_filename}")
            prediction_results = pd.read_csv(inference_filename, header=0, converters={'descriptor': literal_return})

            if output_column not in prediction_results:
                self.logger.error(f"Column {output_column} does not exist in prediction file")
                raise Exception(f"Column {output_column} does not exist in prediction file")

            # get the gt using the task wither all at one query or one-by-one
            if hasattr(gt_processor, 'get_all') and callable(gt_processor.get_all):
                ground_truth = gt_processor.get_all(prediction_results['descriptor'])
            else:
                ground_truth = [gt_processor(sample) for sample in prediction_results['descriptor'].values]

            gt = [FuseUtilsHierarchicalDict.get(x, gt_label_name).item() for x in ground_truth]
            prediction_results['ground_truth'] = gt
            prediction_results.rename(columns={output_column: 'score', 'descriptor': 'key'}, inplace=True)

            if weights_function is not None:
                prediction_results = weights_function(prediction_results)
            else:
                prediction_results['weight'] = 1
                prediction_results['stratum'] = -1

            self.raw_data = prediction_results
        else:
            self.logger.info(f"Reading {len(raw_data)} records from raw_data")
            self.raw_data = raw_data
        pass

    def get_all_keys(self) -> pd.Series:
        return self.raw_data.index

    def get_keys_by_label(self, label: Label) -> pd.Series:
        match_results = label.match_entity(self.raw_data)
        return self.raw_data[match_results].index

    def get_scores_by_label_idx(self, keys, label_idx):
        if 'score' in self.raw_data:
            # binary class
            return self.raw_data.loc[keys].score
        else:
            # multi class case
            return self.raw_data.loc[keys].scores.str[label_idx]

    def get_data(self):
        return self.raw_data


def literal_return(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val
