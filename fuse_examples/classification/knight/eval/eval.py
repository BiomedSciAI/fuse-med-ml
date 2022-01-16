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
import os
from typing import List, Tuple, Union
from collections import OrderedDict

import csv
import pandas as pd
import numpy as np

from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
from fuse.utils.utils_file import FuseUtilsFile

from fuse.analyzer.analyzer_default import FuseAnalyzerDefault

from fuse.metrics.metric_confidence_interval import FuseMetricConfidenceInterval
from fuse.metrics.classification.metric_auc import FuseMetricAUC
from fuse.metrics.classification.metric_roc_curve import FuseMetricROCCurve
from pandas.core.frame import DataFrame


## Constants
# Constants that defines the expected format of the prediction and target files and list the classes for task 1 and task @
EXPECTED_TASK1_PRED_KEYS = {'case_id', 'NoAT-score', 'CanAT-score'}
EXPECTED_TASK2_PRED_KEYS = {'case_id', 'B-score', 'LR-score', 'IR-score', 'HR-score', 'VHR-score'}
EXPECTED_TARGET_KEYS = {'case_id', 'Task1-target', 'Task2-target'}
PRED_SAMPLE_DESC_NAME = "case_id"
TARGET_SAMPLE_DESC_NAME = "case_id"
TASK1_CLASS_NAMES = ("NoAT", "CanAT") # must be aligned with task1 targets
TASK2_CLASS_NAMES = ("B", "LR", "IR", "HR", "VHR") # must be aligned with task2 targets



def post_processing(sample_dict: dict, task1: bool = True, task2: bool = True) -> dict:
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
    set(expected_keys).issubset(set(FuseUtilsHierarchicalDict.get_all_keys(sample_dict)))

    # convert scores to numpy array
    # task 1 
    if task1:
        task1_pred = []
        for cls_name in TASK1_CLASS_NAMES:
            task1_pred.append(FuseUtilsHierarchicalDict.get(sample_dict, f"task1_pred.{cls_name}-score" ))
        task1_pred_array = np.array(task1_pred)
        if not np.isclose(task1_pred_array.sum(), 1.0, rtol=0.05):
            raise Exception(f"Error: expecting task 1 prediction for case {FuseUtilsHierarchicalDict.get(sample_dict, 'descriptor')} to sum up to almost 1.0, got {task1_pred_array}")
        FuseUtilsHierarchicalDict.set(sample_dict, 'task1_pred.array', task1_pred_array)

    # task 2
    if task2:
        task2_pred = []
        for cls_name in TASK2_CLASS_NAMES:
            task2_pred.append(FuseUtilsHierarchicalDict.get(sample_dict, f"task2_pred.{cls_name}-score" ))
        task2_pred_array = np.array(task2_pred)
        if not np.isclose(task2_pred_array.sum(), 1.0, rtol=0.05):
            raise Exception(f"Error: expecting task 2 prediction for case {FuseUtilsHierarchicalDict.get(sample_dict, 'descriptor')} to sum up to almost 1.0, got {task2_pred_array}")
        FuseUtilsHierarchicalDict.set(sample_dict, 'task2_pred.array', task2_pred_array)

    return sample_dict

def decode_results(results: dict, output_dir: str, task1: bool, task2: bool) -> Tuple[OrderedDict, str]:
    """
    Gets the results computed by the dictionary and summarize it in a markdown text and dictionary.
    The dictionary will be saved in <output_dir>/results.csv and the markdown text in <output_dir>/results.md
    :param results: the results computed by the metrics
    :param output_dir: path to an output directory
    :param task1: if true will evaluate task1
    :param task2: if true will evaluate task2
    :return: ordered dict summarizing the results and markdown text
    """
    results_table = OrderedDict()
    # Table
    ## task1
    if task1:
        results_table['Task1-AUC'] = f"{FuseUtilsHierarchicalDict.get(results, 'task1_auc.macro_avg.org'):.3f}"
        results_table['Task1-AUC-CI'] = f"[{FuseUtilsHierarchicalDict.get(results, 'task1_auc.macro_avg.conf_lower'):.3f}-{FuseUtilsHierarchicalDict.get(results, 'task1_auc.macro_avg.conf_upper'):.3f}]"
        
    ## task2
    if task2:
        results_table['Task2-AUC'] = f"{FuseUtilsHierarchicalDict.get(results, 'task2_auc.macro_avg.org'):.3f}"
        results_table['Task2-AUC-CI'] = f"[{FuseUtilsHierarchicalDict.get(results, 'task2_auc.macro_avg.conf_lower'):.3f}-{FuseUtilsHierarchicalDict.get(results, 'task2_auc.macro_avg.conf_upper'):.3f}]"
        for cls_name in TASK2_CLASS_NAMES:
            results_table[f'Task2-AUC-{cls_name}VsRest'] = f"{FuseUtilsHierarchicalDict.get(results, f'task2_auc.{cls_name}.org'):.3f}"
            results_table[f'Task2-AUC-{cls_name}VsRest-CI'] = f"[{FuseUtilsHierarchicalDict.get(results, f'task2_auc.{cls_name}.conf_lower'):.3f}-{FuseUtilsHierarchicalDict.get(results, f'task2_auc.{cls_name}.conf_upper'):.3f}]"


    # mark down text
    results_text = ""
    ## task 1
    if task1:
        results_text += "# Task 1 - adjuvant treatment candidacy classification\n"
        results_text += f"AUC: {results_table['Task1-AUC']} {results_table['Task1-AUC-CI']}\n"
        results_text += "## ROC Curve\n"
        results_text += f'<br/>\n<img src="task1_roc.png" alt="drawing" width="40%"/>\n<br/>\n'
    
    ## task 2
    if task2:
        results_text += "\n# Task 2 - risk categories classification\n"
        results_text += f"AUC: {results_table['Task2-AUC']} {results_table['Task2-AUC-CI']}\n"
        
        results_text += "## Multi-Class AUC\n"
        table_columns = ["AUC"] + [f"AUC-{cls_name}VsRest" for cls_name in TASK2_CLASS_NAMES]
        results_text += "\n|"
        results_text += "".join([f" {column} |" for column in table_columns])
        results_text += "\n|"
        results_text += "".join([f" ------ |" for column in table_columns])
        results_text += "\n|"
        results_text += "".join([f" {results_table[f'Task2-{column}']} {results_table[f'Task2-{column}-CI']} |" for column in table_columns])
        
        results_text += "\n## ROC Curve\n"
        results_text += f'<br/>\n<img src="task2_roc.png" alt="drawing" width="40%"/>\n<br/>\n'


    # save files
    with open(os.path.join(output_dir, "results.md"), "w") as output_file:
        output_file.write(results_text)

    with open(os.path.join(output_dir, "results.csv"), "w") as output_file:
        w = csv.writer(output_file)
        w.writerow(results_table.keys())
        w.writerow(results_table.values())

    return results_table, results_text

def eval(task1_prediction_filename: str, task2_prediction_filename: str, target_filename: str, output_dir: str, samples_descr_source: Union[str, List[str]] = None) -> Tuple[OrderedDict, str]:
    """
    Load the prediction and target files, evaluate the predictions and save the results into files.
    :param task1_prediction_filename: path to a prediction csv file for task1. Expecting the columns listed in EXPECTED_TASK1_PRED_KEYS to exist (including the header).
                                      if set to "" - the script will not evaluate task1 
    :param task2_prediction_filename: path to a prediction csv file for task2. Expecting the columns listed in EXPECTED_TASK2_PRED_KEYS to exist (including the header) 
                                      if set to "" - the script will not evaluate task2
    :param target_filename: path to a prediction csv file. Expecting the columns listed in TARGET_SAMPLE_DESC_NAME to exist
    :param output_dir: path to directory to save the output files
    :param samples_descr_source: will run the evaluation on the specified list of sample descriptors. Supported values:
                                 * "task1_pred" to evaluate all the samples specified in task1 prediction file.
                                 * "task2_pred" to evaluate all the samples specified in task2 prediction file.
                                 * "target" to evaluate all the samples specified in targets file.
                                 * list to define the samples explicitly
    :return: ordered dict summarizing the results and markdown text
    """
    FuseUtilsFile.create_dir(output_dir)
    
    # eval task1, task2 or both
    task1 = task1_prediction_filename is not None and task1_prediction_filename != ""
    task2 = task2_prediction_filename is not None and task2_prediction_filename != ""

    if samples_descr_source is None:
        if task1:
            samples_descr_source = "task1_pred"
        else:
            samples_descr_source = "task2_pred"
            
    dataframes_dict = {}
    metrics = {}
    # task 1
    if task1:
        # metrics to evaluate
        metrics.update({
            "task1_auc": FuseMetricConfidenceInterval(FuseMetricAUC(pred_name='task1_pred.array', target_name='target.Task1-target', class_names=TASK1_CLASS_NAMES), stratum_name="target.Task1-target"),
            "task1_roc_curve": FuseMetricROCCurve(pred_name='task1_pred.array', target_name='target.Task1-target', class_names=[None, ""], output_filename=os.path.join(output_dir, "task1_roc.png")),
        })
        # read files
        task1_pred_df = pd.read_csv(task1_prediction_filename, dtype={PRED_SAMPLE_DESC_NAME: object})
        # verify input
        assert set(task1_pred_df.keys()).issubset(EXPECTED_TASK1_PRED_KEYS), \
            f'Expecting task1 prediction file {os.path.abspath(task1_prediction_filename)} to include also the following keys: {EXPECTED_TASK1_PRED_KEYS - set(task1_pred_df.keys())}'
        task1_pred_df = task1_pred_df.set_index(PRED_SAMPLE_DESC_NAME, drop=False)
        dataframes_dict["task1_pred"] = task1_pred_df

    # task 2
    if task2:
        # metrics to evaluate
        metrics.update({
            "task2_auc": FuseMetricConfidenceInterval(FuseMetricAUC(pred_name='task2_pred.array', target_name='target.Task2-target', class_names=TASK2_CLASS_NAMES), stratum_name="target.Task2-target"),
            "task2_roc_curve": FuseMetricROCCurve(pred_name='task2_pred.array', target_name='target.Task2-target', class_names=TASK2_CLASS_NAMES, output_filename=os.path.join(output_dir, "task2_roc.png")),
        })
        # read files
        task2_pred_df = pd.read_csv(task2_prediction_filename, dtype={PRED_SAMPLE_DESC_NAME: object})
        # verify input
        assert set(task2_pred_df.keys()).issubset(EXPECTED_TASK2_PRED_KEYS), \
            f'Expecting task2 prediction file {os.path.abspath(task2_prediction_filename)} to include also the following keys: {EXPECTED_TASK2_PRED_KEYS - set(task2_pred_df.keys())}'
        task2_pred_df = task2_pred_df.set_index(PRED_SAMPLE_DESC_NAME, drop=False)
        dataframes_dict["task2_pred"] = task2_pred_df
    
    # read files
    target_df = pd.read_csv(target_filename, dtype={TARGET_SAMPLE_DESC_NAME: object})
    # verify input
    assert set(target_df.keys()).issubset(EXPECTED_TARGET_KEYS), \
        f'Expecting target file {os.path.abspath(target_filename)} to include also the following keys: {EXPECTED_TARGET_KEYS - set(target_df.keys())}'
    target_df = target_df.set_index(TARGET_SAMPLE_DESC_NAME, drop=False)
    dataframes_dict["target"] = target_df

    # analyze
    analyzer = FuseAnalyzerDefault()
    results = analyzer.analyze_prediction_dataframe(sample_descr=samples_descr_source, 
                                                    dataframes_dict=dataframes_dict, 
                                                    post_processing=lambda x: post_processing(x, task1, task2),
                                                    metrics=metrics, 
                                                    output_filename=None)

    # output
    return decode_results(results, output_dir=output_dir, task1=task1, task2=task2)


if __name__ == "__main__":
    """
    Run evaluation:
    Usage: python eval.py <target_filename> <task1 prediction_filename> <task2 prediction_filename> <output dir>
    See details in function eval()
    Run dummy example (set the working dir to fuse-med-ml/fuse_examples/classification/knight/eval): python eval.py example/example_targets.csv example/example_task1_predictions.csv example/example_task2_predictions.csv example/results
    """
    if len(sys.argv) == 1:
        # no arguments - set arguments inline - see details in function eval()
        target_filename = "example2/example_targets.csv"
        task1_prediction_filename = None
        task2_prediction_filename = "example2/example_task2_predictions.csv"
        output_dir = "example2/results"
    else:
        # get arguments from sys.argv
        assert len(sys.argv) == 5, f'Error: expecting 4 input arguments, but got {len(sys.argv)-1}. Usage: python eval.py <target_filename> <task1_prediction_filename> <task2_prediction_filename> <output_dir>. See details in function eval()'
        target_filename = sys.argv[1]
        task1_prediction_filename = sys.argv[2]
        task2_prediction_filename = sys.argv[3]
        output_dir = sys.argv[4]
    
    eval(target_filename=target_filename, task1_prediction_filename=task1_prediction_filename, task2_prediction_filename=task2_prediction_filename, output_dir=output_dir)