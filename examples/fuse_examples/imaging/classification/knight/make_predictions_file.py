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
import pathlib

# add parent directory to path, so that 'knight' folder is treated as a module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from typing import Optional, Union
import pandas as pd

from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import save_dataframe
from fuse.dl.managers.manager_default import ManagerDefault

from examples.fuse_examples.imaging.classification.knight.eval.eval import TASK1_CLASS_NAMES, TASK2_CLASS_NAMES
from baseline.dataset import knight_dataset


def make_predictions_file(
    model_dir: str,
    checkpoint: str,
    data_path: str,
    cache_path: Optional[str],
    split: Union[str, dict],
    output_filename: str,
    predictions_key_name: str,
    task_num: int,
):
    """
    Automaitically make prediction files in the requested format - given path to model dir create by FuseMedML during training
    :param model_dir: path to model dir create by FuseMedML during training
    :param data_path: path to the original data downloaded from https://github.com/neheller/KNIGHT
    :param cache_path: Optional - path to the cache folder. If none, it will pre-processes the data again
    :param split: either path to pickled dictionary or the actual dictionary specifing the split between train and validation. the dictionary maps "train" to list of sample descriptors and "val" to list of sample descriptions
    :param output_filename: filename of the output csv file
    :param predictions_key_name: the key in batch_dict of the model predictions
    :param task_num: either 1 or 2 (task 1 or task 2)
    """
    # Logger
    fuse_logger_start(console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("KNIGHT: make predictions file in FuseMedML", {"attrs": ["bold", "underline"]})
    lgr.info(f"predictions_filename={os.path.abspath(output_filename)}", {"color": "magenta"})

    # Data
    # read train/val splits file.
    if isinstance(split, str):
        split = pd.read_pickle(split)
        if isinstance(split, list):
            # For this example, we use split 0 out of the the available cross validation splits
            split = split[0]
    if split is None:  # test mode
        json_filepath = os.path.join(data_path, "features.json")
        data = pd.read_json(json_filepath)
        split = {"test": list(data.case_id)}

    _, validation_dl, test_dl, _, _, _ = knight_dataset(data_path, cache_path, split, reset_cache=False, batch_size=2)

    if "test" in split:
        dl = test_dl
    else:
        dl = validation_dl

    # Manager for inference
    manager = ManagerDefault()
    predictions_df = manager.infer(
        data_loader=dl,
        input_model_dir=model_dir,
        checkpoint=checkpoint,
        output_columns=[predictions_key_name],
        output_file_name=None,
    )

    # Convert to required format
    if task_num == 1:
        class_names = TASK1_CLASS_NAMES
    elif task_num == 2:
        class_names = TASK2_CLASS_NAMES
    else:
        raise Exception(f"Unexpected task num {task_num}")
    predictions_score_names = [f"{cls_name}-score" for cls_name in class_names]
    predictions_df[predictions_score_names] = pd.DataFrame(
        predictions_df[predictions_key_name].tolist(), index=predictions_df.index
    )
    predictions_df.reset_index(inplace=True)
    predictions_df.rename({"index": "case_id"}, axis=1, inplace=True)
    predictions_df = predictions_df[["case_id"] + predictions_score_names]

    # save file
    save_dataframe(predictions_df, output_filename, index=False)


if __name__ == "__main__":
    """
    Automaitically make prediction files in the requested format - given path to model dir create by FuseMedML during training
    Usage: python make_predictions_file <model_dir> <checkpint> <data_path> <cache_path> <split_path> <output_filename> <predictions_key_name> <task_num>.
    See details in function הmake_predictions_file.
    """
    if len(sys.argv) == 1:
        # no arguments - set arguments inline - see details in function make_predictions_file
        model_dir = ""
        checkpoint = "best"
        data_path = ""
        cache_path = ""
        split = f"{pathlib.Path(__file__).parent.resolve()}/baseline/splits_final.pkl"
        output_filename = "validation_predictions.csv"
        predictions_key_name = "model.output.head_0"
        task_num = 1  # 1 or 2
    else:
        # get arguments from sys.argv
        assert (
            len(sys.argv) == 8
        ), "Error: expecting 8 arguments. Usage: python make_predictions_file <model_dir> <checkpint> <data_path> <cache_path> <split_path> <output_filename> <predictions_key_name>. See details in function make_predictions_file."
        model_dir = sys.argv[1]
        checkpoint = sys.argv[2]
        data_path = sys.argv[3]
        cache_path = sys.argv[4]
        split = sys.argv[5]
        output_filename = sys.argv[6]
        predictions_key_name = sys.argv[7]
        task_num = sys.argv[8]

    make_predictions_file(
        model_dir=model_dir,
        checkpoint=checkpoint,
        data_path=data_path,
        cache_path=cache_path,
        split=split,
        output_filename=output_filename,
        predictions_key_name=predictions_key_name,
        task_num=task_num,
    )
