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

import sys
import logging
import os
from typing import Optional, Union
import pandas as pd

from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.utils_file import FuseUtilsFile
from fuse.managers.manager_default import FuseManagerDefault

from fuse_examples.classification.knight.eval.eval import TASK1_CLASS_NAMES, TASK2_CLASS_NAMES 

from KNIGHT.dataset import knight_dataset

def make_predictions_file(model_dir: str, 
                          checkpoint: str, 
                          data_path: str, 
                          cache_path: Optional[str], 
                          split: Union[str, dict],
                          output_filename: str, 
                          predictions_name: str="model.output.head_0"):
    """
    Automaitically make prediction files in the requested format - given path to model dir create by FuseMedML during training
    :param model_dir: path to model dir create by FuseMedML during training
    :param data_path: path to the original data downloaded from https://github.com/neheller/KNIGHT
    :param cache_path: Optional - path to the cache folder. If none, it will pre-processes the data again
    :param split: either path to pickled dictionary or the actual dictionary specifing the split between train and validation. the dictionary maps "train" to list of sample descriptors and "val" to list of sample descriptions 
    :param output_filename: filename of the output csv file
    :param predictions_name: the key in batch_dict of the model predictions 
    """
    # Logger
    fuse_logger_start(console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('KNIGHT: make predictions file in FuseMedML', {'attrs': ['bold', 'underline']})
    lgr.info(f'predictions_filename={os.path.abspath(output_filename)}', {'color': 'magenta'})

    # Data
    # read train/val splits file.
    if isinstance(split, str): 
        split=pd.read_pickle(split)
        if isinstance(split, list):
            # For this example, we use split 0 out of the the available cross validation splits
            split = split[0]
   
    _, validation_dl  = knight_dataset(data_path, cache_path, split, reset_cache=False, batch_size=2)

    # Manager for inference
    manager = FuseManagerDefault()
    predictions_df = manager.infer(data_loader=validation_dl,
                  input_model_dir=model_dir,
                  checkpoint=checkpoint,
                  output_columns=[predictions_name],
                  output_file_name=None)

    
    # Convert to required format
    predictions_score_names = [f"{cls_name}-score" for cls_name in TASK1_CLASS_NAMES]
    predictions_df[predictions_score_names] = pd.DataFrame(predictions_df[predictions_name].tolist(), index=predictions_df.index)
    predictions_df.reset_index(inplace=True)
    predictions_df.rename({"descriptor": "case_id"}, axis=1, inplace=True)
    predictions_df = predictions_df[["case_id"] + predictions_score_names]

    # save file
    FuseUtilsFile.save_dataframe(predictions_df, output_filename, index=False)   

if __name__ == "__main__":
    """
    Automaitically make prediction files in the requested format - given path to model dir create by FuseMedML during training
    Usage: python make_predictions_file <model_dir> <checkpint> <data_path> <cache_path> <split_path> <output_filename> <predictions_key_name>. 
    See details in function make_predictions_file.
    """
    if len(sys.argv) == 1:
        # no arguments - set arguments inline - see details in function make_predictions_file
        model_dir = ""
        checkpoint = "best"
        data_path = ""
        cache_path = ""
        split = "baseline/splits_final.pkl"
        output_filename = "validation_predictions.csv"
        predictions_key_name = "model.output.head_0"
    else:
        # get arguments from sys.argv
        assert len(sys.argv) == 8, "Error: expecting 8 arguments. Usage: python make_predictions_file <model_dir> <checkpint> <data_path> <cache_path> <split_path> <output_filename> <predictions_key_name>. See details in function make_predictions_file."
        model_dir = sys.argv[1]
        checkpoint = sys.argv[2]
        data_path = sys.argv[3]
        cache_path = sys.argv[4]
        split = sys.argv[5]
        output_filename = sys.argv[6]
        predictions_key_name = sys.argv[7]

    make_predictions_file(model_dir=model_dir, checkpoint=checkpoint, data_path=data_path, cache_path=cache_path, split=split, output_filename=output_filename, predictions_key_name=predictions_key_name)