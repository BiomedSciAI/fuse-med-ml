
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
EXPECTED_PRED_KEYS = {'case_id', 'Task1-NoAT-score', 'Task1-CanAT-score', 'Task2-B-score', 'Task2-LR-score', 'Task2-IR-score', 'Task2-HR-score', 'Task2-VHR-score'}
EXPECTED_TARGET_KEYS = {'case_id', 'Task1-target', 'Task2-target'}
PRED_SAMPLE_DESC_NAME = "case_id"
TARGET_SAMPLE_DESC_NAME = "case_id"
TASK1_CLASS_NAMES = ("NoAT", "CanAT") # must be aligned with task1 targets
TASK2_CLASS_NAMES = ("B", "LR", "IR", "HR", "VHR") # must be aligned with task2 targets



def post_processing(sample_dict: dict) -> dict:
    """
    post caching processing. Will group together to an array the per class scores and verify it sums up to 1.0
    :param sample_dict: a dictionary that contais all the extracted values of a single sample
    :return: a modified/alternative dictionary
    """
    # verify sample
    expected_keys = [f"pred.{key}" for key in EXPECTED_PRED_KEYS]
    expected_keys += [f"target.{key}" for key in EXPECTED_TARGET_KEYS]
    set(expected_keys).issubset(set(FuseUtilsHierarchicalDict.get_all_keys(sample_dict)))

    # convert scores to numpy array
    # task 1 
    pred_task1 = []
    for cls_name in TASK1_CLASS_NAMES:
        pred_task1.append(FuseUtilsHierarchicalDict.get(sample_dict, f"pred.Task1-{cls_name}-score" ))
    pred_task1_array = np.array(pred_task1)
    if not np.isclose(pred_task1_array.sum(), 1.0, rtol=0.05):
        raise Exception(f"Error: expecting task 1 prediction for case {FuseUtilsHierarchicalDict.get(sample_dict, 'descriptor')} to sun up to almost 1.0, got {pred_task1_array}")
    FuseUtilsHierarchicalDict.set(sample_dict, 'pred.task1', pred_task1_array)

    # task 2
    pred_task2 = []
    for cls_name in TASK2_CLASS_NAMES:
        pred_task2.append(FuseUtilsHierarchicalDict.get(sample_dict, f"pred.Task2-{cls_name}-score" ))
    pred_task2_array = np.array(pred_task2)
    FuseUtilsHierarchicalDict.set(sample_dict, 'pred.task2', pred_task2_array)

    return sample_dict

def decode_results(results: dict, output_dir: str) -> Tuple[OrderedDict, str]:
    """
    Gets the results computed by the dictionary and summarize it in a markdown text and dictionary.
    The dictionary will be saved in <output_dir>/results.csv and the markdown text in <output_dir>/results.md
    :param results: the results computed by the metrics
    :param output_dir: path to an output directory
    :return: ordered dict summarizing the results and markdown text
    """
    results_table = OrderedDict()
    # Table
    ## task1
    results_table['Task1-AUC'] = f"{FuseUtilsHierarchicalDict.get(results, 'task1_auc.macro_avg.org'):.2f}"
    results_table['Task1-AUC-CI'] = f"[{FuseUtilsHierarchicalDict.get(results, 'task1_auc.macro_avg.conf_lower'):.2f}-{FuseUtilsHierarchicalDict.get(results, 'task1_auc.macro_avg.conf_upper'):.2f}]"
    
    ## task2
    results_table['Task2-AUC'] = f"{FuseUtilsHierarchicalDict.get(results, 'task2_auc.macro_avg.org'):.2f}"
    results_table['Task2-AUC-CI'] = f"[{FuseUtilsHierarchicalDict.get(results, 'task2_auc.macro_avg.conf_lower'):.2f}-{FuseUtilsHierarchicalDict.get(results, 'task2_auc.macro_avg.conf_upper'):.2f}]"
    for cls_name in TASK2_CLASS_NAMES:
        results_table[f'Task2-AUC-{cls_name}VsRest'] = f"{FuseUtilsHierarchicalDict.get(results, f'task2_auc.{cls_name}.org'):.2f}"
        results_table[f'Task2-AUC-{cls_name}VsRest-CI'] = f"[{FuseUtilsHierarchicalDict.get(results, f'task2_auc.{cls_name}.conf_lower'):.2f}-{FuseUtilsHierarchicalDict.get(results, f'task2_auc.{cls_name}.conf_upper'):.2f}]"


    # mark down text
    results_text = ""
    ## task 1
    results_text += "# Task 1 - adjuvant treatment candidacy classification\n"
    results_text += f"AUC: {results_table['Task1-AUC']} {results_table['Task1-AUC-CI']}\n"
    results_text += "## ROC Curve\n"
    results_text += f'<br/>\n<img src="task1_roc.png" alt="drawing" width="20%"/>\n<br/>\n'
    
    ## task 2
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
    results_text += f'<br/>\n<img src="task2_roc.png" alt="drawing" width="20%"/>\n<br/>\n'


    # save files
    with open(os.path.join(output_dir, "results.md"), "w") as output_file:
        output_file.write(results_text)

    with open(os.path.join(output_dir, "results.csv"), "w") as output_file:
        w = csv.writer(output_file)
        w.writerow(results_table.keys())
        w.writerow(results_table.values())

    return results_table, results_text

def eval(prediction_filename: str, target_filename: str, output_dir: str, samples_descr_source: Union[str, List[str]] = "pred") -> Tuple[OrderedDict, str]:
    """
    Load the prediction and target files, evaluate the predictions and save the results into files.
    :param prediction_filename: path to a prediction csv file. Expecting the columns listed in PRED_SAMPLE_DESC_NAME to exist 
    :param target_filename: path to a prediction csv file. Expecting the columns listed in TARGET_SAMPLE_DESC_NAME to exist
    :param output_dir: path to directory to save the output files
    :param samples_descr_source: will run the evaluation on the specified list of sample descriptors. Supported values:
                                 * "pred" to evaluate all the samples specified in prediction file.
                                 * "target" to evaluate all the samples specified in targets file.
                                 * list to define the samples explicitly
    :return: ordered dict summarizing the results and markdown text
    """
    FuseUtilsFile.create_or_reset_dir(output_dir, force_reset=True)
    # metrics to evaluate
    metrics = {
        # task 1
        "task1_auc": FuseMetricConfidenceInterval(FuseMetricAUC(pred_name='pred.task1', target_name='target.Task1-target', class_names=TASK1_CLASS_NAMES), stratum_name="target.Task1-target"),
        "task1_roc_curve": FuseMetricROCCurve(pred_name='pred.task1', target_name='target.Task1-target', class_names=[None, ""], output_filename=os.path.join(output_dir, "task1_roc.png")),
        # task 2
        "task2_auc": FuseMetricConfidenceInterval(FuseMetricAUC(pred_name='pred.task2', target_name='target.Task2-target', class_names=TASK2_CLASS_NAMES), stratum_name="target.Task2-target"),
        "task2_roc_curve": FuseMetricROCCurve(pred_name='pred.task2', target_name='target.Task2-target', class_names=TASK2_CLASS_NAMES, output_filename=os.path.join(output_dir, "task2_roc.png")),
    }
    
    # read files
    pred_df = pd.read_csv(prediction_filename, dtype={PRED_SAMPLE_DESC_NAME: object})
    target_df = pd.read_csv(target_filename, dtype={TARGET_SAMPLE_DESC_NAME: object})

    # verify input
    assert set(pred_df.keys()).issubset(EXPECTED_PRED_KEYS), \
        f'Expecting prediction file {os.path.abspath(prediction_filename)} to include also the following keys: {EXPECTED_PRED_KEYS - set(pred_df.keys())}'
    assert set(target_df.keys()).issubset(EXPECTED_TARGET_KEYS), \
        f'Expecting target file {os.path.abspath(target_filename)} to include also the following keys: {EXPECTED_TARGET_KEYS - set(target_df.keys())}'

    # analyze
    pred_df = pred_df.set_index(PRED_SAMPLE_DESC_NAME, drop=False)
    target_df = target_df.set_index(TARGET_SAMPLE_DESC_NAME, drop=False)
    analyzer = FuseAnalyzerDefault()
    results = analyzer.analyze_prediction_dataframe(sample_descr=samples_descr_source, 
                                                    dataframes_dict={'pred': pred_df, 'target': target_df}, 
                                                    post_processing=post_processing,
                                                    metrics=metrics, 
                                                    output_filename=None)

    # output
    return decode_results(results, output_dir=output_dir)


if __name__ == "__main__":
    """
    Rum evaluation:
    Usage: python eval.py <target_filename> <prediction_filename> <output dir>
    Run dummy example (set the working dir to fuse-med-ml/fuse_examples/classification/knight/eval): python eval.py example_targets.csv example_predictions.csv ./example_output_dir
    """
    assert len(sys.argv) == 4, f'Error: expecting 3 input arguments, but got {len(sys.argv)-1}. Usage: python eval.py <target_filename> <prediction_filename> <output dir>'
    eval(target_filename=sys.argv[1], prediction_filename=sys.argv[2], output_dir=sys.argv[3])