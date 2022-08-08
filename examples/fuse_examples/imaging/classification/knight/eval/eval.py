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
import pathlib
import sys
import os
from typing import List, Tuple, Union, Optional
from collections import OrderedDict

import csv
import pandas as pd
import numpy as np

from fuse.utils.ndict import NDict
from fuse.utils.file_io.file_io import create_dir

from fuse.eval.evaluator import EvaluatorDefault

from fuse.eval.metrics.classification.metrics_classification_common import (
    MetricAUCROC,
    MetricROCCurve,
)
from fuse.eval.metrics.metrics_common import CI
from functools import partial

## Constants
# Constants that defines the expected format of the prediction and target files and list the classes for task 1 and task @
EXPECTED_TASK1_PRED_KEYS = {"case_id", "NoAT-score", "CanAT-score"}
EXPECTED_TASK2_PRED_KEYS = {"case_id", "B-score", "LR-score", "IR-score", "HR-score", "VHR-score"}
EXPECTED_TARGET_KEYS = {"case_id", "Task1-target", "Task2-target"}
PRED_CASE_ID_NAME = "case_id"
TARGET_CASE_ID_NAME = "case_id"
TASK1_CLASS_NAMES = ("NoAT", "CanAT")  # must be aligned with task1 targets
TASK2_CLASS_NAMES = ("B", "LR", "IR", "HR", "VHR")  # must be aligned with task2 targets


def post_processing(sample_dict: NDict, task1: bool = True, task2: bool = True) -> dict:
    """
    post caching processing. Will group together to an array the per class scores and verify it sums up to 1.0
    :param sample_dict: a dictionary that contais all the extracted values of a single sample
    :param task1: if true will evaluate task1
    :param task2: if true will evaluate task2
    :return: a modified/alternative dictionary
    """
    # verify sample
    expected_keys = [f"target.{key}" for key in EXPECTED_TARGET_KEYS]
    if task1:
        expected_keys += [f"task1_pred.{key}" for key in EXPECTED_TASK1_PRED_KEYS]
    if task2:
        expected_keys += [f"task2_pred.{key}" for key in EXPECTED_TASK2_PRED_KEYS]
    set(expected_keys).issubset(set(sample_dict.keypaths()))

    # convert scores to numpy array
    # task 1
    if task1:
        task1_pred = []
        for cls_name in TASK1_CLASS_NAMES:
            task1_pred.append(sample_dict[f"task1_pred.{cls_name}-score"])
        task1_pred_array = np.array(task1_pred)
        if not np.isclose(task1_pred_array.sum(), 1.0, rtol=0.05):
            print(
                f"Warning: expecting task 1 prediction for case {sample_dict['descriptor']} to sum up to almost 1.0, got {task1_pred_array}"
            )
        sample_dict["task1_pred.array"] = task1_pred_array

    # task 2
    if task2:
        task2_pred = []
        for cls_name in TASK2_CLASS_NAMES:
            task2_pred.append(sample_dict[f"task2_pred.{cls_name}-score"])
        task2_pred_array = np.array(task2_pred)
        if not np.isclose(task2_pred_array.sum(), 1.0, rtol=0.05):
            print(
                f"Error: expecting task 2 prediction for case {sample_dict['descriptor']} to sum up to almost 1.0, got {task2_pred_array}"
            )
        sample_dict["task2_pred.array"] = task2_pred_array

    return sample_dict


def decode_results(results: NDict, output_dir: str, task1: bool, task2: bool) -> Tuple[OrderedDict, str]:
    """
    Gets the results computed by the dictionary and summarize it in a markdown text and dictionary.
    The dictionary will be saved in <output_dir>/results.csv and the markdown text in <output_dir>/results.md
    :param results: the results computed by the metrics
    :param output_dir: path to an output directory
    :param task1: if true will evaluate task1
    :param task2: if true will evaluate task2
    :return: ordered dict summarizing the results and markdown text
    """
    results = NDict(results["metrics"])
    results_table = OrderedDict()
    # Table
    ## task1
    if task1:
        results_table["Task1-AUC"] = f"{results['task1_auc.macro_avg.org']:.3f}"
        results_table[
            "Task1-AUC-CI"
        ] = f"[{results['task1_auc.macro_avg.conf_lower']:.3f}-{results['task1_auc.macro_avg.conf_upper']:.3f}]"

    ## task2
    if task2:
        results_table["Task2-AUC"] = f"{results['task2_auc.macro_avg.org']:.3f}"
        results_table[
            "Task2-AUC-CI"
        ] = f"[{results['task2_auc.macro_avg.conf_lower']:.3f}-{results['task2_auc.macro_avg.conf_upper']:.3f}]"
        for cls_name in TASK2_CLASS_NAMES:
            results_table[f"Task2-AUC-{cls_name}VsRest"] = f"{results[f'task2_auc.{cls_name}.org']:.3f}"
            results_table[
                f"Task2-AUC-{cls_name}VsRest-CI"
            ] = f"[{results[f'task2_auc.{cls_name}.conf_lower']:.3f}-{results[f'task2_auc.{cls_name}.conf_upper']:.3f}]"

    # mark down text
    results_text = ""
    ## task 1
    if task1:
        results_text += "# Task 1 - adjuvant treatment candidacy classification\n"
        results_text += f"AUC: {results_table['Task1-AUC']} {results_table['Task1-AUC-CI']}\n"
        results_text += "## ROC Curve\n"
        results_text += '<br/>\n<img src="task1_roc.png" alt="drawing" width="40%"/>\n<br/>\n'

    ## task 2
    if task2:
        results_text += "\n# Task 2 - risk categories classification\n"
        results_text += f"AUC: {results_table['Task2-AUC']} {results_table['Task2-AUC-CI']}\n"

        results_text += "## Multi-Class AUC\n"
        table_columns = ["AUC"] + [f"AUC-{cls_name}VsRest" for cls_name in TASK2_CLASS_NAMES]
        results_text += "\n|"
        results_text += "".join([f" {column} |" for column in table_columns])
        results_text += "\n|"
        results_text += "".join([" ------ |" for column in table_columns])
        results_text += "\n|"
        results_text += "".join(
            [f" {results_table[f'Task2-{column}']} {results_table[f'Task2-{column}-CI']} |" for column in table_columns]
        )
        results_text += "\n## ROC Curve\n"
        results_text += '<br/>\n<img src="task2_roc.png" alt="drawing" width="40%"/>\n<br/>\n'

    # save files
    with open(os.path.join(output_dir, "results.md"), "w") as output_file:
        output_file.write(results_text)

    with open(os.path.join(output_dir, "results.csv"), "w") as output_file:
        w = csv.writer(output_file)
        w.writerow(results_table.keys())
        w.writerow(results_table.values())

    return results_table, results_text


def eval(
    task1_prediction_filename: str,
    task2_prediction_filename: str,
    target_filename: str,
    output_dir: str,
    case_ids_source: Optional[Union[str, List[str]]] = "target",
) -> Tuple[OrderedDict, str]:
    """
    Load the prediction and target files, evaluate the predictions and save the results into files.
    :param task1_prediction_filename: path to a prediction csv file for task1. Expecting the columns listed in EXPECTED_TASK1_PRED_KEYS to exist (including the header).
                                      if set to "" - the script will not evaluate task1
    :param task2_prediction_filename: path to a prediction csv file for task2. Expecting the columns listed in EXPECTED_TASK2_PRED_KEYS to exist (including the header)
                                      if set to "" - the script will not evaluate task2
    :param target_filename: path to a prediction csv file. Expecting the columns listed in TARGET_CASE_ID_NAME to exist
    :param output_dir: path to directory to save the output files
    :param case_ids_source: will run the evaluation on the specified list of case ids. Supported values:
                                 * "task1_pred" to evaluate all the samples/cases specified in task1 prediction file.
                                 * "task2_pred" to evaluate all the samples/cases specified in task2 prediction file.
                                 * "target" to evaluate all the samples/cases specified in targets file.
                                 * list to define the samples explicitly
    :return: ordered dict summarizing the results and markdown text
    """
    create_dir(output_dir)

    # eval task1, task2 or both
    task1 = task1_prediction_filename is not None and task1_prediction_filename != ""
    task2 = task2_prediction_filename is not None and task2_prediction_filename != ""

    if case_ids_source is None:
        if task1:
            case_ids_source = "task1_pred"
        else:
            case_ids_source = "task2_pred"

    dataframes_dict = {}
    metrics = {}
    post_proc = partial(post_processing, task1=task1, task2=task2)
    # task 1
    if task1:
        # metrics to evaluate
        metrics.update(
            {
                "task1_auc": CI(
                    MetricAUCROC(
                        pred="task1_pred.array",
                        target="target.Task1-target",
                        class_names=TASK1_CLASS_NAMES,
                        pre_collect_process_func=post_proc,
                    ),
                    stratum="target.Task1-target",
                ),
                "task1_roc_curve": MetricROCCurve(
                    pred="task1_pred.array",
                    target="target.Task1-target",
                    class_names=[None, ""],
                    pre_collect_process_func=post_proc,
                    output_filename=os.path.join(output_dir, "task1_roc.png"),
                ),
            }
        )
        # read files
        task1_pred_df = pd.read_csv(task1_prediction_filename, dtype={PRED_CASE_ID_NAME: object})
        # verify input
        assert set(task1_pred_df.keys()).issubset(
            EXPECTED_TASK1_PRED_KEYS
        ), f"Expecting task1 prediction file {os.path.abspath(task1_prediction_filename)} to include also the following keys: {EXPECTED_TASK1_PRED_KEYS - set(task1_pred_df.keys())}"
        task1_pred_df["id"] = task1_pred_df[PRED_CASE_ID_NAME]
        dataframes_dict["task1_pred"] = task1_pred_df

    # task 2
    if task2:
        # metrics to evaluate
        metrics.update(
            {
                "task2_auc": CI(
                    MetricAUCROC(
                        pred="task2_pred.array",
                        target="target.Task2-target",
                        class_names=TASK2_CLASS_NAMES,
                        pre_collect_process_func=post_proc,
                    ),
                    stratum="target.Task2-target",
                ),
                "task2_roc_curve": MetricROCCurve(
                    pred="task2_pred.array",
                    target="target.Task2-target",
                    class_names=TASK2_CLASS_NAMES,
                    output_filename=os.path.join(output_dir, "task2_roc.png"),
                    pre_collect_process_func=post_proc,
                ),
            }
        )
        # read files
        task2_pred_df = pd.read_csv(task2_prediction_filename, dtype={PRED_CASE_ID_NAME: object})
        # verify input
        assert set(task2_pred_df.keys()).issubset(
            EXPECTED_TASK2_PRED_KEYS
        ), f"Expecting task2 prediction file {os.path.abspath(task2_prediction_filename)} to include also the following keys: {EXPECTED_TASK2_PRED_KEYS - set(task2_pred_df.keys())}"
        task2_pred_df["id"] = task2_pred_df[PRED_CASE_ID_NAME]
        dataframes_dict["task2_pred"] = task2_pred_df

    # read files
    target_df = pd.read_csv(target_filename, dtype={TARGET_CASE_ID_NAME: object})
    # verify input
    assert set(target_df.keys()).issubset(
        EXPECTED_TARGET_KEYS
    ), f"Expecting target file {os.path.abspath(target_filename)} to include also the following keys: {EXPECTED_TARGET_KEYS - set(target_df.keys())}"
    target_df["id"] = target_df[TARGET_CASE_ID_NAME]
    dataframes_dict["target"] = target_df

    # analyze
    evaluator = EvaluatorDefault()
    results = evaluator.eval(
        ids=list(dataframes_dict[case_ids_source]["id"]), data=dataframes_dict, metrics=metrics, output_dir=None
    )

    # output
    return decode_results(results, output_dir=output_dir, task1=task1, task2=task2)


if __name__ == "__main__":
    """
    Run evaluation:
    Usage: python eval.py <target_filename> <task1 prediction_filename> <task2 prediction_filename> <output dir>
    See details in function eval()
    Run dummy example (set the working dir to fuse-med-ml/examples/fuse_examples/imaging/classification/knight/eval): python eval.py example/example_targets.csv example/example_task1_predictions.csv example/example_task2_predictions.csv example/results
    """
    if len(sys.argv) == 1:
        dir_path = pathlib.Path(__file__).parent.resolve()
        # no arguments - set arguments inline - see details in function eval()
        target_filename = os.path.join(dir_path, "example/example_targets.csv")
        task1_prediction_filename = os.path.join(dir_path, "example/example_task1_predictions.csv")
        task2_prediction_filename = os.path.join(dir_path, "example/example_task2_predictions.csv")
        output_dir = "example/result"
    else:
        # get arguments from sys.argv
        assert (
            len(sys.argv) == 5
        ), f"Error: expecting 4 input arguments, but got {len(sys.argv)-1}. Usage: python eval.py <target_filename> <task1_prediction_filename> <task2_prediction_filename> <output_dir>. See details in function eval()"
        target_filename = sys.argv[1]
        task1_prediction_filename = sys.argv[2]
        task2_prediction_filename = sys.argv[3]
        output_dir = sys.argv[4]

    eval(
        target_filename=target_filename,
        task1_prediction_filename=task1_prediction_filename,
        task2_prediction_filename=task2_prediction_filename,
        output_dir=output_dir,
    )
