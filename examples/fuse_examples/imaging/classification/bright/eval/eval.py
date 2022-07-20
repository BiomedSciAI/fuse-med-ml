import pathlib
import sys
import os
from typing import List, Tuple, Union
from collections import OrderedDict

import csv
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.utils.ndict import NDict
import pandas as pd
import numpy as np

from fuse.utils.file_io.file_io import create_or_reset_dir

from fuse.eval.evaluator import EvaluatorDefault
from fuse.eval.metrics.classification.metrics_classification_common import MetricBSS, MetricConfusion
from fuse.eval.metrics.metrics_common import CI


## Constants
# Constants that defines the expected format of the prediction and target files and list the classes for task 1 and task @
EXPECTED_TASK1_PRED_KEYS = {
    "image_name",
    "predicted_label",
    "Noncancerous-score",
    "Precancerous-score",
    "Cancerous-score",
}
EXPECTED_TASK2_PRED_KEYS = {
    "image_name",
    "predicted_label",
    "PB-score",
    "UDH-score",
    "FEA-score",
    "ADH-score",
    "DCIS-score",
    "IC-score",
}
EXPECTED_TARGET_KEYS = {"image_name", "Task1-target", "Task2-target"}
PRED_SAMPLE_DESC_NAME = "image_name"
TARGET_SAMPLE_DESC_NAME = "image_name"
TASK1_CLASS_NAMES = ("Noncancerous", "Precancerous", "Cancerous")  # must be aligned with task1 targets
TASK2_CLASS_NAMES = ("PB", "UDH", "FEA", "ADH", "DCIS", "IC")  # must be aligned with task2 targets


def process(sample_dict: NDict) -> dict:
    """
    post caching processing. Will group together to an array the per class scores and verify it sums up to 1.0
    :param sample_dict: a dictionary that contais all the extracted values of a single sample
    :return: a modified/alternative dictionary
    """
    # verify sample
    expected_keys = [f"task1_pred.{key}" for key in EXPECTED_TASK1_PRED_KEYS]
    expected_keys += [f"task2_pred.{key}" for key in EXPECTED_TASK2_PRED_KEYS]
    expected_keys += [f"target.{key}" for key in EXPECTED_TARGET_KEYS]
    set(expected_keys).issubset(set(sample_dict.keypaths()))

    # convert scores to numpy array
    # task 1
    task1_pred = []
    for cls_name in TASK1_CLASS_NAMES:
        task1_pred.append(sample_dict[f"task1_pred.{cls_name}-score"])
    task1_pred_array = np.array(task1_pred)
    if not np.isclose(task1_pred_array.sum(), 1.0, rtol=0.05):
        raise Exception(
            f"Error: expecting task 1 prediction for case {sample_dict['descriptor']} to sum up to almost 1.0, got {task1_pred_array}"
        )
    sample_dict["task1_pred.array"] = task1_pred_array

    # task 2
    task2_pred = []
    for cls_name in TASK2_CLASS_NAMES:
        task2_pred.append(sample_dict[f"task2_pred.{cls_name}-score"])
    task2_pred_array = np.array(task2_pred)
    if not np.isclose(task2_pred_array.sum(), 1.0, rtol=0.05):
        raise Exception(
            f"Error: expecting task 2 prediction for case {sample_dict['descriptor']} to sum up to almost 1.0, got {task2_pred_array}"
        )
    sample_dict["task2_pred.array"] = task2_pred_array

    return sample_dict


def decode_results(results: dict, output_dir: str) -> Tuple[OrderedDict, str]:
    """
    Gets the results computed by the dictionary and summarize it in a markdown text and dictionary.
    The dictionary will be saved in <output_dir>/results.csv and the markdown text in <output_dir>/results.md
    :param results: the results computed by the metrics
    :param output_dir: path to an output directory
    :return: ordered dict summarizing the results and markdown text
    """
    results = NDict(results["metrics"])
    results_table = OrderedDict()
    # Table
    ## task1
    results_table["Task1-F1"] = f"{results['task1_f1.f1.macro_avg.org']:.3f}"
    results_table[
        "Task1-F1-CI"
    ] = f"[{results['task1_f1.f1.macro_avg.conf_lower']:.3f}-{results['task1_f1.f1.macro_avg.conf_upper']:.3f}]"
    for cls_name in TASK1_CLASS_NAMES:
        results_table[f"Task1-F1-{cls_name}VsRest"] = f"{results[f'task1_f1.f1.{cls_name}.org']:.3f}"
        results_table[
            f"Task1-F1-{cls_name}VsRest-CI"
        ] = f"[{results[f'task1_f1.f1.{cls_name}.conf_lower']:.3f}-{results[f'task1_f1.f1.{cls_name}.conf_upper']:.3f}]"
    results_table["Task1-BSS"] = f"{results['task1_bss.org']:.3f}"
    results_table["Task1-BSS-CI"] = f"[{results['task1_bss.conf_lower']:.3f}-{results['task1_bss.conf_upper']:.3f}]"

    ## task2
    results_table["Task2-F1"] = f"{results['task2_f1.f1.macro_avg.org']:.3f}"
    results_table[
        "Task2-F1-CI"
    ] = f"[{results['task2_f1.f1.macro_avg.conf_lower']:.3f}-{results['task2_f1.f1.macro_avg.conf_upper']:.3f}]"
    for cls_name in TASK2_CLASS_NAMES:
        results_table[f"Task2-F1-{cls_name}VsRest"] = f"{results[f'task2_f1.f1.{cls_name}.org']:.3f}"
        results_table[
            f"Task2-F1-{cls_name}VsRest-CI"
        ] = f"[{results[f'task2_f1.f1.{cls_name}.conf_lower']:.3f}-{results[f'task2_f1.f1.{cls_name}.conf_upper']:.3f}]"
    results_table["Task2-BSS"] = f"{results['task2_bss.org']:.3f}"
    results_table["Task2-BSS-CI"] = f"[{results['task2_bss.conf_lower']:.3f}-{results['task2_bss.conf_upper']:.3f}]"

    # mark down text
    results_text = ""
    ## task 1
    results_text += "# Task 1 - 3-class WSI classification\n"
    results_text += f"F1: {results_table['Task1-F1']} {results_table['Task1-F1-CI']}\n"

    results_text += "## Multi-Class F1\n"
    table_columns = ["F1"] + [f"F1-{cls_name}VsRest" for cls_name in TASK1_CLASS_NAMES]
    results_text += "\n|"
    results_text += "".join([f" {column} |" for column in table_columns])
    results_text += "\n|"
    results_text += "".join([" ------ |" for column in table_columns])
    results_text += "\n|"
    results_text += "".join(
        [f" {results_table[f'Task1-{column}']} {results_table[f'Task1-{column}-CI']} |" for column in table_columns]
    )

    results_text += "\n## BSS\n"
    results_text += f"BSS: {results_table['Task1-BSS']} {results_table['Task1-BSS-CI']}\n"

    ## task 2
    results_text += "\n# Task 2 - 6-class WSI classification\n"
    results_text += f"F1: {results_table['Task2-F1']} {results_table['Task2-F1-CI']}\n"

    results_text += "## Multi-Class F1\n"
    table_columns = ["F1"] + [f"F1-{cls_name}VsRest" for cls_name in TASK2_CLASS_NAMES]
    results_text += "\n|"
    results_text += "".join([f" {column} |" for column in table_columns])
    results_text += "\n|"
    results_text += "".join([" ------ |" for column in table_columns])
    results_text += "\n|"
    results_text += "".join(
        [f" {results_table[f'Task2-{column}']} {results_table[f'Task2-{column}-CI']} |" for column in table_columns]
    )

    results_text += "\n## BSS\n"
    results_text += f"BSS: {results_table['Task2-BSS']} {results_table['Task2-BSS-CI']}\n"

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
    samples_descr_source: Union[str, List[str]] = "task1_pred",
) -> Tuple[OrderedDict, str]:
    """
    Load the prediction and target files, evaluate the predictions and save the results into files.
    :param task1_prediction_filename: path to a prediction csv file for task1. Expecting the columns listed in EXPECTED_TASK1_PRED_KEYS to exist (including the header)
    :param task2_prediction_filename: path to a prediction csv file for task2. Expecting the columns listed in EXPECTED_TASK2_PRED_KEYS to exist (including the header)
    :param target_filename: path to a prediction csv file. Expecting the columns listed in EXPECTED_TARGET_KEYS to exist
    :param output_dir: path to directory to save the output files
    :param samples_descr_source: will run the evaluation on the specified list of sample descriptors. Supported values:
                                 * "task1_pred" to evaluate all the samples specified in task1 prediction file.
                                 * "task2_pred" to evaluate all the samples specified in task2 prediction file.
                                 * "target" to evaluate all the samples specified in targets file.
                                 * list to define the samples explicitly
    :return: ordered dict summarizing the results and markdown text
    """
    create_or_reset_dir(output_dir, force_reset=True)

    # metrics to evaluate
    metrics = OrderedDict(
        [
            # task 1
            ("task1_op", MetricApplyThresholds(pred="task1_pred.predicted_label")),  # will apply argmax
            (
                "task1_f1",
                CI(
                    MetricConfusion(
                        pred="results:metrics.task1_op.cls_pred",
                        target="target.Task1-target",
                        class_names=TASK1_CLASS_NAMES,
                        metrics=("f1",),
                        pre_collect_process_func=process,
                    ),
                    stratum="target.Task1-target",
                ),
            ),
            (
                "task1_bss",
                CI(
                    MetricBSS(pred="task1_pred.array", target="target.Task1-target", pre_collect_process_func=process),
                    stratum="target.Task1-target",
                ),
            ),
            # task 2
            ("task2_op", MetricApplyThresholds(pred="task2_pred.predicted_label")),  # will apply argmax
            (
                "task2_f1",
                CI(
                    MetricConfusion(
                        pred="results:metrics.task1_op.cls_pred",
                        target="target.Task2-target",
                        class_names=TASK2_CLASS_NAMES,
                        metrics=("f1",),
                        pre_collect_process_func=process,
                    ),
                    stratum="target.Task2-target",
                ),
            ),
            (
                "task2_bss",
                CI(
                    MetricBSS(pred="task2_pred.array", target="target.Task2-target", pre_collect_process_func=process),
                    stratum="target.Task2-target",
                ),
            ),
        ]
    )
    # read files
    task1_pred_df = pd.read_csv(task1_prediction_filename, dtype={PRED_SAMPLE_DESC_NAME: object})
    task2_pred_df = pd.read_csv(task2_prediction_filename, dtype={PRED_SAMPLE_DESC_NAME: object})
    target_df = pd.read_csv(target_filename, dtype={TARGET_SAMPLE_DESC_NAME: object})

    # verify input
    assert set(task1_pred_df.keys()).issubset(
        EXPECTED_TASK1_PRED_KEYS
    ), f"Expecting task1 prediction file {os.path.abspath(task1_prediction_filename)} to include also the following keys: {EXPECTED_TASK1_PRED_KEYS - set(task1_pred_df.keys())}"
    assert set(task2_pred_df.keys()).issubset(
        EXPECTED_TASK2_PRED_KEYS
    ), f"Expecting task2 prediction file {os.path.abspath(task2_prediction_filename)} to include also the following keys: {EXPECTED_TASK2_PRED_KEYS - set(task2_pred_df.keys())}"
    assert set(target_df.keys()).issubset(
        EXPECTED_TARGET_KEYS
    ), f"Expecting target file {os.path.abspath(target_filename)} to include also the following keys: {EXPECTED_TARGET_KEYS - set(target_df.keys())}"

    # analyze
    task1_pred_df["id"] = task1_pred_df["image_name"]
    task2_pred_df["id"] = task2_pred_df["image_name"]
    target_df["id"] = target_df["image_name"]
    evaluator = EvaluatorDefault()
    results = evaluator.eval(
        ids=list(target_df["id"]),
        data={"task1_pred": task1_pred_df, "task2_pred": task2_pred_df, "target": target_df},
        metrics=metrics,
        output_dir=None,
    )

    # output
    return decode_results(results, output_dir=output_dir)


if __name__ == "__main__":
    """
    Run evaluation:
    Usage: python eval.py <target_filename> <task1_prediction_filename> <task2_prediction_filename> <output dir>
    Run dummy example (set the working dir to fuse-med-ml/examples/fuse_examples/imaging/classification/bright/eval): python eval.py example/example_targets.csv example/example_task1_predictions.csv example/example_task2_predictions.csv example/results
    Run baseline (set the working dir to fuse-med-ml/examples/fuse_examples/imaging/classification/bright/eval): python eval.py validation_targets.csv baseline/validation_baseline_task1_predictions.csv baseline/validation_baseline_task2_predictions.csv baseline/validation_results
    """

    if len(sys.argv) == 1:
        dir_path = pathlib.Path(__file__).parent.resolve()
        target_filename = os.path.join(dir_path, "validation_targets.csv")
        task1_prediction_filename = os.path.join(dir_path, "baseline/validation_baseline_task1_predictions.csv")
        task2_prediction_filename = os.path.join(dir_path, "baseline/validation_baseline_task2_predictions.csv")
        output_dir = os.path.join(dir_path, "baseline/validation_results")
        eval(
            target_filename=target_filename,
            task1_prediction_filename=task1_prediction_filename,
            task2_prediction_filename=task2_prediction_filename,
            output_dir=output_dir,
        )
    else:
        assert (
            len(sys.argv) == 5
        ), f"Error: expecting 4 input arguments, but got {len(sys.argv)-1}. Usage: python eval.py <target_filename> <task1_prediction_filename> <task2_prediction_filename> <output dir>"
        eval(
            target_filename=sys.argv[1],
            task1_prediction_filename=sys.argv[2],
            task2_prediction_filename=sys.argv[3],
            output_dir=sys.argv[4],
        )
