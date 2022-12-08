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

from typing import Union
import pandas as pd

from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import save_dataframe


def make_targets_file(data_path: str, split: Union[str, dict], output_filename: str):
    """
    Automaitically make targets file in the requested format
    :param data_path: path to the original data downloaded from https://github.com/neheller/KNIGHT
    :param cache_path: Optional - path to the cache folder. If none, it will pre-processes the data again
    :param split: either path to pickled dictionary or the actual dictionary specifing the split between train and validation. the dictionary maps "train" to list of sample descriptors and "val" to list of sample descriptions
    :param output_filename: filename of the output csv file
    """
    # Logger
    fuse_logger_start(console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("KNIGHT: make targets file", {"attrs": ["bold", "underline"]})
    lgr.info(f"targets_filename={os.path.abspath(output_filename)}", {"color": "magenta"})

    # Data
    # read train/val splits file.
    # use validation set if split specified, otherwise assume testset
    if split is not None:
        is_validation_set = True
    else:
        assert split is None, f"Error: unexpected split format {split}"
        is_validation_set = False

    if is_validation_set:
        if isinstance(split, str):
            split = pd.read_pickle(split)
        if isinstance(split, list):
            # For this example, we use split 0 out of the the available cross validation splits
            split = split[0]

        json_labels_filepath = os.path.join(data_path, "knight.json")
        labels = pd.read_json(json_labels_filepath)
        labels = labels[labels["case_id"].isin(split["val"])]
        labels = pd.DataFrame({"target": labels["aua_risk_group"].values})

    else:  # test mode
        # if this function is ran in test mode, then presumably the user has the test labels
        # for test purposes, the test labels file isn't shared with participants:
        json_labels_filepath = os.path.join(data_path, "knight_test_labels.json")
        if not os.path.isfile(json_labels_filepath):
            ValueError("No test labels file found")
        labels = pd.read_json(json_labels_filepath, typ="series")
        labels = pd.DataFrame({"target": labels.values})

    levels = ["benign", "low_risk", "intermediate_risk", "high_risk", "very_high_risk"]
    labels["Task2-target"] = labels["target"].apply(lambda x: levels.index(x))
    labels.insert(1, "Task1-target", labels["Task2-target"].apply(lambda x: int(x > 3)))
    labels.reset_index(inplace=True)
    labels.rename({"index": "case_id"}, axis=1, inplace=True)
    labels.drop(["target"], axis=1, inplace=True)
    save_dataframe(labels, output_filename, index=False)
    return


if __name__ == "__main__":
    """
    Automaitically make targets file (csv files that holds just the labels for tasks) in the requested format
    Usage: python make_predictions_file <data_path> <cache_path> <split> <output_filename>
    See details in function make_predictions_file.
    """
    # no arguments - set arguments inline - see details in function make_targets_file
    data_path = ""
    split = "baseline/splits_final.pkl"
    output_filename = "validation_targets.csv"
    make_targets_file(data_path=data_path, split=split, output_filename=output_filename)
