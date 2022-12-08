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

import os
import pathlib
from tempfile import mkdtemp
from typing import Any, Dict

from collections import OrderedDict

import pandas as pd
import numpy as np
from fuse.eval.metrics.survival.metrics_survival import MetricCIndex

from fuse.utils import set_seed

from fuse.eval.metrics.metrics_common import GroupAnalysis, CI, Filter
from fuse.eval.metrics.metrics_model_comparison import PairedBootstrap
from fuse.eval.metrics.classification.metrics_classification_common import (
    MetricAUCPR,
    MetricAUCROC,
    MetricAccuracy,
    MetricConfusion,
    MetricConfusionMatrix,
    MetricBSS,
    MetricROCCurve,
)
from fuse.eval.metrics.classification.metrics_model_comparison_common import (
    MetricDelongsTest,
    MetricMcnemarsTest,
)
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_calibration_common import (
    MetricReliabilityDiagram,
    MetricECE,
    MetricFindTemperature,
    MetricApplyTemperature,
)
from fuse.eval.metrics.classification.metrics_ensembling_common import MetricEnsemble
from fuse.eval.evaluator import EvaluatorDefault


def example_0() -> Dict[str, Any]:
    """
    Simple evaluation example for binary classification task
    Input is a single dataframe pickle file including 3 columns: "id", "pred" (numpy arrays) and "target"
    """
    # path to prediction and target files
    dir_path = pathlib.Path(__file__).parent.resolve()
    input_filename = os.path.join(dir_path, "inputs/example0.pickle")

    # list of metrics
    metrics = OrderedDict(
        [
            ("auc", MetricAUCROC(pred="pred", target="target")),
        ]
    )

    # read files
    data = input_filename
    evaluator = EvaluatorDefault()
    results = evaluator.eval(
        ids=None, data=data, metrics=metrics
    )  # ids == None -> run evaluation on all available samples

    return results


def example_1() -> Dict[str, Any]:
    """
    Simple evaluation example
    Inputs are two .csv files: one including predictions and one targets
    The predictions requires a simple pre-processing to create prediction in a format of vector of probabilities
    """
    # path to prediction and target files
    dir_path = pathlib.Path(__file__).parent.resolve()
    prediction_filename = os.path.join(dir_path, "inputs/example1_predictions.csv")
    targets_filename = os.path.join(dir_path, "inputs/example1_targets.csv")

    # define data
    data = {"pred": prediction_filename, "target": targets_filename}

    # pre collect function to change the format
    def pre_collect_process(sample_dict: dict) -> dict:
        # convert scores to numpy array
        task1_pred = []
        for cls_name in ("NoAT", "CanAT"):
            task1_pred.append(sample_dict[f"pred.{cls_name}-score"])
        task1_pred_array = np.array(task1_pred)
        sample_dict["pred.array"] = task1_pred_array

        return sample_dict

    # list of metrics
    metrics = OrderedDict(
        [
            (
                "auc",
                MetricAUCROC(
                    pred="pred.array", target="target.Task1-target", pre_collect_process_func=pre_collect_process
                ),
            ),
        ]
    )

    # specify just the list you want to evaluate - predictions may contain more samples that will be ignored
    task_target_df = pd.read_csv(targets_filename)
    ids = list(task_target_df["id"])

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=ids, data=data, metrics=metrics)

    return results


def example_2():
    """
    Cross validation example - evaluation the entire data, built from few folds at once
    Multiple inference files - each include prediction of a different fold - binary predictions (single probability)
    Single target file with labels for all the folds
    """
    # path to prediction and target files
    dir_path = pathlib.Path(__file__).parent.resolve()
    prediction_fold0_filename = os.path.join(dir_path, "inputs/example2_predictions_fold0.csv")
    prediction_fold1_filename = os.path.join(dir_path, "inputs/example2_predictions_fold1.csv")
    targets_filename = os.path.join(dir_path, "inputs/example2_targets.csv")

    # define data
    data = {"pred": [prediction_fold0_filename, prediction_fold1_filename], "target": targets_filename}

    # list of metrics
    metrics = OrderedDict(
        [
            ("auc", MetricAUCROC(pred="pred.CanAT-score", target="target.Task1-target")),
            (
                "auc_per_fold",
                GroupAnalysis(
                    MetricAUCROC(pred="pred.CanAT-score", target="target.Task1-target"), group="pred.evaluator_fold"
                ),
            ),
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data, metrics=metrics)

    return results


def example_3():
    """
    General group analysis example - compute the AUC for each group separately.
    In this case the grouping is done according to gender
    """
    data = {
        "pred": [0.1, 0.2, 0.6, 0.7, 0.8, 0.3, 0.6, 0.2, 0.7, 0.9],
        "target": [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
        "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "gender": ["female", "female", "female", "female", "female", "male", "male", "male", "male", "male"],
    }
    data = pd.DataFrame(data)

    metrics = OrderedDict(
        [("auc_per_group", GroupAnalysis(MetricAUCROC(pred="pred", target="target"), group="gender"))]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data, metrics=metrics)

    return results


def example_4() -> Dict[str, Any]:
    """
    Simple evaluation example with Confidence Interval
    Inputs are two .csv files: one including predictions and one targets
    """
    # set seed
    seed = 1234
    # path to prediction and target files - reuse example1 inputs
    dir_path = pathlib.Path(__file__).parent.resolve()
    prediction_filename = os.path.join(dir_path, "inputs/example1_predictions.csv")
    targets_filename = os.path.join(dir_path, "inputs/example1_targets.csv")

    # define data
    data = {"pred": prediction_filename, "target": targets_filename}

    # list of metrics
    metrics = OrderedDict(
        [
            (
                "auc",
                CI(
                    MetricAUCROC(pred="pred.CanAT-score", target="target.Task1-target"),
                    stratum="target.Task1-target",
                    rnd_seed=seed,
                ),
            ),
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data, metrics=metrics)

    return results


def example_5():
    """
    Model comparison using paired bootstrap metric
    Compare model a binary classification sensitivity to model b binary classification sensitivity
    """
    # set seed
    seed = 0

    # define data
    data = {
        "id": [0, 1, 2, 3, 4],
        "model_a_pred": [0.4, 0.3, 0.7, 0.8, 0.0],
        "model_b_pred": [0.4, 0.3, 0.7, 0.2, 1.0],
        "target": [0, 1, 1, 1, 0],
    }

    data_df = pd.DataFrame(data)

    # list of metrics
    metric_model_test = MetricConfusion(pred="results:metrics.apply_thresh_a.cls_pred", target="target")
    metric_model_reference = MetricConfusion(pred="results:metrics.apply_thresh_b.cls_pred", target="target")

    metrics = OrderedDict(
        [
            ("apply_thresh_a", MetricApplyThresholds(pred="model_a_pred", operation_point=0.5)),
            ("apply_thresh_b", MetricApplyThresholds(pred="model_b_pred", operation_point=0.5)),
            (
                "compare_a_to_b",
                PairedBootstrap(
                    metric_model_test,
                    metric_model_reference,
                    stratum="target",
                    metric_keys_to_compare=["sensitivity"],
                    rnd_seed=seed,
                ),
            ),
        ]
    )

    # read files
    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data_df, metrics=metrics)

    return results


def example_6() -> Dict:
    """
    Simple test of the DeLong's test implementation
    Also "naively" test the multiclass mode (one vs. all) by simply extending the
    binary classifier prediction vectors into a 2D matrix
    For this simple example, DeLong's test was computed manually in this blog post by Rachel Draelos:
    https://glassboxmedicine.com/2020/02/04/comparing-aucs-of-machine-learning-models-with-delongs-test/
    """
    target = np.array([0, 0, 1, 1, 1])
    pred1 = np.array([0.1, 0.2, 0.6, 0.7, 0.8])
    pred2 = np.array([0.3, 0.6, 0.2, 0.7, 0.9])

    # convert to Nx2 for multi-class generalization:
    pred1 = np.stack((1 - pred1, pred1), 1)
    pred2 = np.stack((1 - pred2, pred2), 1)
    index = np.arange(0, len(target))

    # convert to dataframes:
    pred_df = pd.DataFrame(columns=["pred1", "pred2", "id"])
    pred_df["pred1"] = list(pred1)
    pred_df["pred2"] = list(pred2)
    pred_df["id"] = index
    target_df = pd.DataFrame(columns=["target", "id"])
    target_df["target"] = target
    target_df["id"] = index
    data = {"pred": pred_df, "target": target_df}

    # list of metrics
    metrics = OrderedDict(
        [
            (
                "delongs_test",
                MetricDelongsTest(
                    target="target.target", class_names=["negative", "positive"], pred1="pred.pred1", pred2="pred.pred2"
                ),
            ),
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data, metrics=metrics)

    return results


def example_7() -> Dict:
    """
    Another example for testing the DeLong's test implementation. This time in "binary classifier" mode
    The sample data in this example was used in the above blog post and verified against an R implementation.
    Three different sources: dataframe per model prediction and a target dataframe
    """
    target = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    pred1 = np.array([0.1, 0.2, 0.05, 0.3, 0.1, 0.6, 0.6, 0.7, 0.8, 0.99, 0.8, 0.67, 0.5])
    pred2 = np.array([0.3, 0.6, 0.2, 0.1, 0.1, 0.9, 0.23, 0.7, 0.9, 0.4, 0.77, 0.3, 0.89])

    ids = np.arange(0, len(target))

    # convert to dataframes:
    pred1_df = pd.DataFrame(columns=["pred1", "pred2", "id"])
    pred1_df["output"] = pred1
    pred1_df["id"] = ids

    pred2_df = pd.DataFrame(columns=["pred1", "pred2", "id"])
    pred2_df["output"] = pred2
    pred2_df["id"] = ids

    target_df = pd.DataFrame(columns=["target", "id"])
    target_df["target"] = target
    target_df["id"] = ids
    data = {"pred1": pred1_df, "pred2": pred2_df, "target": target_df}

    # list of metrics
    metrics = OrderedDict(
        [
            ("delongs_test", MetricDelongsTest(target="target.target", pred1="pred1.output", pred2="pred2.output")),
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data, metrics=metrics)

    return results


def example_8():
    """
    Classification Multiclass example: five classes evaluation with metrics AUC-ROC AUC-PR, sensitivity, specificity and precision
    Input: one .csv prediction file that requires processing to convert the predictions to numpy array and one target file.
    """
    # path to prediction and target files
    dir_path = pathlib.Path(__file__).parent.resolve()
    prediction_filename = os.path.join(dir_path, "inputs/example7_predictions.csv")
    targets_filename = os.path.join(dir_path, "inputs/example1_targets.csv")  # same target file as in example1
    data = {"target": targets_filename, "pred": prediction_filename}

    class_names = ["B", "LR", "IR", "HR", "VHR"]

    # pre collect function to change the format
    def pre_collect_process(sample_dict: dict) -> dict:
        # convert scores to numpy array
        task2_pred = []
        for cls_name in class_names:
            task2_pred.append(sample_dict[f"pred.{cls_name}-score"])
        task2_pred_array = np.array(task2_pred)
        sample_dict["pred.output"] = task2_pred_array
        sample_dict["pred.output_class"] = task2_pred_array.argmax()

        return sample_dict

    # list of metrics
    metrics = OrderedDict(
        [
            (
                "auc",
                MetricAUCROC(
                    pred="pred.output",
                    target="target.Task2-target",
                    class_names=class_names,
                    pre_collect_process_func=pre_collect_process,
                ),
            ),
            (
                "auc_pr",
                MetricAUCPR(
                    pred="pred.output",
                    target="target.Task2-target",
                    class_names=class_names,
                    pre_collect_process_func=pre_collect_process,
                ),
            ),
            (
                "confusion",
                MetricConfusion(
                    pred="pred.output_class",
                    target="target.Task2-target",
                    class_names=class_names,
                    pre_collect_process_func=pre_collect_process,
                    metrics=["sensitivity", "specificity", "precision"],
                ),
            ),  # default operation point is argmax
            (
                "accuracy",
                MetricAccuracy(
                    pred="pred.output_class", target="target.Task2-target", pre_collect_process_func=pre_collect_process
                ),
            ),  # default operation point is argmax
            (
                "confusion_matrix",
                MetricConfusionMatrix(
                    cls_pred="pred.output_class",
                    target="target.Task2-target",
                    class_names=class_names,
                    pre_collect_process_func=pre_collect_process,
                ),
            ),
            (
                "bss",
                MetricBSS(
                    pred="pred.output", target="target.Task2-target", pre_collect_process_func=pre_collect_process
                ),
            ),
            (
                "roc",
                MetricROCCurve(
                    pred="pred.output",
                    target="target.Task2-target",
                    class_names=class_names,
                    output_filename="roc.png",
                    pre_collect_process_func=pre_collect_process,
                ),
            ),
        ]
    )

    # read files
    evaluator = EvaluatorDefault()
    results = evaluator.eval(
        ids=None, data=data, metrics=metrics
    )  # ids == None -> run evaluation on all available samples

    return results


def example_9():
    """
    Classification example with single-process iterator.
    This example requires fuse-med-ml-data and torchvision packages installed
    """
    import torchvision

    # set seed
    set_seed(1234)

    # Create dataset
    # mnist download path - MNIST_DATA_PATH defined in cicd pipeline
    mnist_data_path = os.environ.get("MNIST_DATA_PATH", mkdtemp(prefix="mnist"))
    torch_dataset = torchvision.datasets.MNIST(mnist_data_path, download=True, train=False)

    # define iterator
    def data_iter():
        for sample_index, (image, label) in enumerate(torch_dataset):
            sample_dict = {}
            sample_dict["id"] = sample_index
            sample_dict["pred"] = np.random.randint(0, 10)
            sample_dict["label"] = label
            yield sample_dict

        # list of metrics

    metrics = OrderedDict(
        [
            (
                "accuracy",
                MetricAccuracy(pred="pred", target="label"),
            ),  # operation_point=None -> no need to convert pred from probabilities to class
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(
        ids=None, data=data_iter(), batch_size=5, metrics=metrics
    )  # ids == None -> run evaluation on all available samples

    return results


def example_10() -> Dict:
    """
    Test of McNemar's test implementation
    """

    # Toy example:

    # The correct target (class) labels (optional)
    ground_truth = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # in case ground_truth is not provided (variable set to None),
    # the significance test checks the similarity between model predictions only.

    # Class labels predicted by model 1
    cls_pred1 = [0, 1, 1, 1, 0, 1, 1, 0, 1, 1]
    # Class labels predicted by model 2
    cls_pred2 = [1, 1, 0, 1, 0, 1, 1, 0, 0, 1]

    index = np.arange(0, len(cls_pred1))
    # convert to dataframes:
    df = pd.DataFrame(columns=["cls_pred1", "cls_pred2", "id"])
    df["cls_pred1"] = cls_pred1
    df["cls_pred2"] = cls_pred2
    df["ground_truth"] = ground_truth
    df["id"] = index
    data = {"data": df}

    # list of metrics
    metrics = OrderedDict(
        [
            (
                "mcnemars_test",
                MetricMcnemarsTest(pred1="data.cls_pred1", pred2="data.cls_pred2", target="data.ground_truth"),
            ),
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data, metrics=metrics)

    return results


def example_11() -> Dict:
    """
    Sub group analysis example
    """
    # define data
    data = {
        "id": [0, 1, 2, 3, 4],
        "pred": [0, 1, 0, 0, 0],
        "gender": ["male", "female", "female", "male", "female"],
        "target": [0, 1, 1, 1, 0],
    }
    data_df = pd.DataFrame(data)

    def pre_collect_process(sample_dict: dict) -> dict:
        sample_dict["filter"] = sample_dict["gender"] != "male"
        return sample_dict

    acc = MetricAccuracy(pred="pred", target="target")
    metrics = OrderedDict(
        [
            (
                "accuracy",
                Filter(acc, "filter", pre_collect_process_func=pre_collect_process),
            ),  # operation_point=None -> no need to convert pred from probabilities to class
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(
        ids=None, data=data_df, metrics=metrics
    )  # ids == None -> run evaluation on all available samples

    return results


def example_12() -> Dict:
    """
    Example of a metric pipeline which includes a per-sample metric/operation.
    First, we apply a simple thresholding operation (per sample "metric"/operation) to generate class predictions.
    Then, we apply the accuracy metric on the class predictions vs. targets.
    """
    target = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    pred = np.array([0.3, 0.6, 0.2, 0.1, 0.1, 0.9, 0.23, 0.7, 0.9, 0.4, 0.77, 0.3, 0.89])

    ids = np.arange(0, len(target))

    # convert to dataframes:
    pred_df = pd.DataFrame(columns=["pred", "id"])
    pred_df["pred"] = pred
    pred_df["id"] = ids

    target_df = pd.DataFrame(columns=["target", "id"])
    target_df["target"] = target
    target_df["id"] = ids
    data = {"pred": pred_df, "target": target_df}

    metrics = OrderedDict(
        [
            ("apply_thresh", MetricApplyThresholds(pred="pred.pred", operation_point=0.5)),
            ("acc", MetricAccuracy(pred="results:metrics.apply_thresh.cls_pred", target="target.target")),
        ]
    )
    # read files
    evaluator = EvaluatorDefault()
    results = evaluator.eval(
        ids=None, data=data, metrics=metrics
    )  # ids == None -> run evaluation on all available samples

    return results


def example_13() -> Dict:
    """
    Test reliability diagram and ECE metrics
    We use multi-class input data as in example_7:
    One .csv prediction file that requires processing to convert the predictions to numpy array and one target file.

    We define a metric pipeline. First the reliability diagram is computed on the original predictions.
    Then, we compute the expected calibration error (ECE). Then, we find an optimal "temperature" value to calibrate the logits.
    Then, we apply this temperature scale on the original logits, to obtain calibrated ones.
    Finally, we compute the ECE and reliability diagram for the calibrated predictions.
    """
    # path to prediction and target files
    dir_path = pathlib.Path(__file__).parent.resolve()
    prediction_filename = os.path.join(dir_path, "inputs/example7_predictions.csv")
    targets_filename = os.path.join(dir_path, "inputs/example1_targets.csv")  # same target file as in example1
    data = {"target": targets_filename, "pred": prediction_filename}

    class_names = ["B", "LR", "IR", "HR", "VHR"]
    num_bins = 10
    num_quantiles = None

    # pre collect function to change the format
    def pre_collect_process(sample_dict: dict) -> dict:
        # convert scores to numpy array
        task2_pred = []
        for cls_name in class_names:
            task2_pred.append(sample_dict[f"pred.{cls_name}-score"])
        task2_pred_array = np.array(task2_pred)
        logit_array = np.log(task2_pred_array)  # "make up" logits up to a constant
        sample_dict["pred.output"] = task2_pred_array
        sample_dict["pred.logits"] = logit_array

        return sample_dict

    # list of metrics
    metrics = OrderedDict(
        [
            (
                "reliability",
                MetricReliabilityDiagram(
                    pred="pred.output",
                    target="target.Task2-target",
                    num_bins=num_bins,
                    num_quantiles=num_quantiles,
                    output_filename="reliability.png",
                    pre_collect_process_func=pre_collect_process,
                ),
            ),
            (
                "ece",
                MetricECE(
                    pred="pred.output",
                    target="target.Task2-target",
                    num_bins=num_bins,
                    num_quantiles=num_quantiles,
                    pre_collect_process_func=pre_collect_process,
                ),
            ),
            (
                "find_temperature",
                MetricFindTemperature(
                    pred="pred.logits", target="target.Task2-target", pre_collect_process_func=pre_collect_process
                ),
            ),
            (
                "apply_temperature",
                MetricApplyTemperature(
                    pred="pred.logits",
                    temperature="results:metrics.find_temperature",
                    pre_collect_process_func=pre_collect_process,
                ),
            ),
            (
                "ece_calibrated",
                MetricECE(
                    pred="results:metrics.apply_temperature",
                    target="target.Task2-target",
                    num_bins=num_bins,
                    num_quantiles=num_quantiles,
                    pre_collect_process_func=pre_collect_process,
                ),
            ),
            (
                "reliability_calibrated",
                MetricReliabilityDiagram(
                    pred="results:metrics.apply_temperature",
                    target="target.Task2-target",
                    num_bins=num_bins,
                    num_quantiles=num_quantiles,
                    output_filename="reliability_calibrated.png",
                    pre_collect_process_func=pre_collect_process,
                ),
            ),
        ]
    )

    # read files
    evaluator = EvaluatorDefault()
    results = evaluator.eval(
        ids=None, data=data, metrics=metrics
    )  # ids == None -> run evaluation on all available samples

    return results


def example_14() -> Dict[str, Any]:
    """
    Model ensemble example
    """
    # path to prediction and target files
    dir_path = pathlib.Path(__file__).parent.resolve()
    inference_file_name = "test_set_infer.gz"
    model_dirs = [
        os.path.join(dir_path, "inputs/ensemble/mnist/test_dir", str(i), inference_file_name) for i in range(5)
    ]
    output_file = "ensemble_output.gz"
    # define data
    data = {str(k): model_dirs[k] for k in range(len(model_dirs))}

    # list of metrics
    metrics = OrderedDict(
        [
            (
                "ensemble",
                MetricEnsemble(
                    pred_keys=[
                        "0.model.output.classification",
                        "1.model.output.classification",
                        "2.model.output.classification",
                        "3.model.output.classification",
                        "4.model.output.classification",
                    ],
                    target="0.data.label",
                    output_file=output_file,
                ),
            ),
            (
                "apply_thresh",
                MetricApplyThresholds(pred="results:metrics.ensemble.preds", operation_point=None),
            ),
            (
                "accuracy",
                MetricAccuracy(
                    pred="results:metrics.apply_thresh.cls_pred",
                    target="0.data.label",
                ),
            ),
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data, metrics=metrics)

    return results


def example_15():
    """
    General group analysis example - compute the AUC for each group separately.
    In this case the grouping is done according to gender
    """
    data = {
        "pred": [0.1, 0.2, 0.6, 0.7, 0.8, 0.3, 0.6, 0.2, 0.7, 0.9],
        "event_times": [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
        "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "gender": ["female", "female", "female", "female", "female", "male", "male", "male", "male", "male"],
    }
    data = pd.DataFrame(data)

    metrics = OrderedDict(
        [("cindex_per_group", GroupAnalysis(MetricCIndex(pred="pred", event_times="event_times"), group="gender"))]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data, metrics=metrics)

    return results


def example_16():
    """
    General group analysis example - compute the AUC for each group separately.
    In this case the grouping is done according to gender
    """
    data = {
        "pred": [0.1, 0.2, 0.6, 0.7, 0.8, 0.3, 0.6, 0.2, 0.7, 0.9],
        "event_observed": 1 - np.asarray([0, 0, 1, 1, 1, 0, 0, 1, 1, 1]),
        "event_times": [1] * 10,
        "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "gender": ["female", "female", "female", "female", "female", "male", "male", "male", "male", "male"],
    }
    data = pd.DataFrame(data)

    metrics = OrderedDict(
        [
            (
                "cindex_per_group",
                GroupAnalysis(
                    MetricCIndex(pred="pred", event_times="event_times", event_observed="event_observed"),
                    group="gender",
                ),
            )
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data, metrics=metrics)

    return results


def example_17() -> Dict:
    """
    Test of C-Index test implementation
    """

    # Toy example:

    # An indicator whether the event was observed [optional: if not supplied all events are assumed to be observed]
    toy_data = dict(
        id=list(range(10)),
        pred=[0.1, 0.2, 0.3, 0.4, 0.5] + [0.2, 0.1, 0.4, 0.3, 0.7],
        event_observed=[0, 0, 0, 0, 0] + [1, 1, 1, 1, 1],
        event_times=[1, 2, 3, 4, 5] + [1, 2, 3, 4, 5],  # event time / censor time
    )
    # convert to dataframes:
    df = pd.DataFrame(data=toy_data)

    data = {"data": df}

    # list of metrics
    metrics = OrderedDict(
        [
            (
                "c_index_test",
                MetricCIndex(pred="data.pred", event_times="data.event_times", event_observed="data.event_observed"),
            ),
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data, metrics=metrics)

    return results
