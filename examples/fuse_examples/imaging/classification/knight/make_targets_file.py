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
from fuse.utils.file_io.file_io import save_dataframe
from fuse.data.utils.export import ExportDataset
from baseline.dataset import knight_dataset

# add parent directory to path, so that 'knight' folder is treated as a module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def make_targets_file(data_path: str, cache_path: Optional[str], split: Union[str, dict], output_filename: str):
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
    if isinstance(split, str):
        is_validation_set = True
    else:
        assert split is None, f"Error: unexpected split format {split}"
        is_validation_set = False

    if is_validation_set:
        split = pd.read_pickle(split)
        if isinstance(split, list):
            # For this example, we use split 0 out of the the available cross validation splits
            split = split[0]
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
        split = {"test": list(labels.keys())}

    _, validation_dl, test_dl, _, _, _ = knight_dataset(
        data_path, cache_path, split, reset_cache=False, only_labels=True
    )

    # Export targets
    if is_validation_set:
        targets_df = ExportDataset.export_to_dataframe(
            validation_dl.dataset,
            ["data.descriptor", "data.gt.gt_global.task_1_label", "data.gt.gt_global.task_2_label"],
        )
    else:  # test set
        targets_df = ExportDataset.export_to_dataframe(
            test_dl.dataset, ["data.descriptor", "data.gt.gt_global.task_1_label", "data.gt.gt_global.task_2_label"]
        )

    # to int and rename keys
    targets_df["data.gt.gt_global.task_1_label"] = targets_df["data.gt.gt_global.task_1_label"].transform(int)
    targets_df["data.gt.gt_global.task_2_label"] = targets_df["data.gt.gt_global.task_2_label"].transform(int)
    targets_df.reset_index(inplace=True)
    targets_df.rename(
        {
            "data.descriptor": "case_id",
            "data.gt.gt_global.task_1_label": "Task1-target",
            "data.gt.gt_global.task_2_label": "Task2-target",
        },
        axis=1,
        inplace=True,
    )

    # save file
    save_dataframe(targets_df, output_filename, index=False)


if __name__ == "__main__":
    """
    Automaitically make targets file (csv files that holds just the labels for tasks) in the requested format
    Usage: python make_predictions_file <data_path> <cache_path> <split> <output_filename>
    See details in function make_predictions_file.
    """
    if len(sys.argv) == 1:
        # no arguments - set arguments inline - see details in function make_targets_file
        data_path = ""
        cache_path = ""
        split = "baseline/splits_final.pkl"
        output_filename = "validation_targets.csv"
    else:
        # get arguments from sys.argv
        assert (
            len(sys.argv) == 8
        ), "Error: expecting 8 arguments. Usage: python make_predictions_file <model_dir> <checkpint> <data_path> <cache_path> <split_path> <output_filename> <predictions_key_name>. See details in function make_predictions_file."
        data_path = sys.argv[1]
        cache_path = sys.argv[2]
        split = sys.argv[3]
        output_filename = sys.argv[4]
    make_targets_file(data_path=data_path, cache_path=cache_path, split=split, output_filename=output_filename)
