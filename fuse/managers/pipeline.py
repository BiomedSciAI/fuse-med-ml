from typing import Dict, Callable
from fuse.utils import gpu as FuseUtilsGPU
from fuse.utils.utils_debug import FuseDebug
from sklearn.model_selection import KFold
import multiprocessing
from functools import partial
from multiprocessing import Process, Queue
from typing import Sequence
import numpy as np
import os
from fuse.eval.metrics.classification.metrics_ensembling_common import MetricEnsemble
from collections import OrderedDict
from fuse.eval.evaluator import EvaluatorDefault
from fuse.utils.file_io.file_io import create_or_reset_dir
from fuse.utils.rand.seed import Seed


# pre collect function to change the format
def ensemble_pre_collect(sample_dict: dict) -> dict:
    # convert predictions from all models to numpy array
    num_classes = sample_dict["0"]["model"]["output"]["classification"].shape[0]
    model_names = list(sample_dict.keys())
    model_names.remove("id")
    pred_array = np.zeros((len(model_names), num_classes))
    for i, m in enumerate(model_names):
        pred_array[i, :] = sample_dict[m]["model"]["output"]["classification"]
    sample_dict["preds"] = pred_array
    sample_dict["target"] = sample_dict["0"]["data"]["label"]

    return sample_dict


def ensemble(test_dirs, test_infer_filename, ensembled_output_file):
    ensembled_output_dir = os.path.dirname(ensembled_output_file)
    create_or_reset_dir(ensembled_output_dir, force_reset=True)
    test_infer_filenames = [os.path.join(d, test_infer_filename) for d in test_dirs]
    # define data for ensemble metric
    data = {str(k): test_infer_filenames[k] for k in range(len(test_infer_filenames))}

    # list of metrics
    metrics = OrderedDict(
        [
            (
                "ensemble",
                MetricEnsemble(
                    preds="preds",
                    target="target",
                    output_file=ensembled_output_file,
                    pre_collect_process_func=ensemble_pre_collect,
                ),
            ),
        ]
    )

    evaluator = EvaluatorDefault()
    _ = evaluator.eval(ids=None, data=data, metrics=metrics)


def runner_wrapper(q_resources, rep_index, fs, *f_args, **f_kwargs):
    rand_gen = Seed.set_seed(rep_index, deterministic_mode=True)
    f_kwargs["rep_index"] = rep_index
    f_kwargs["rand_gen"] = rand_gen
    resource = q_resources.get()
    print(f"Using GPUs: {resource}")
    FuseUtilsGPU.choose_and_enable_multiple_gpus(len(resource), force_gpus=list(resource))
    if isinstance(fs, Sequence):
        for f, last_arg in zip(fs, f_args[-1]):
            f(*(f_args[:-1] + (last_arg,)), **f_kwargs)
    else:
        f(*f_args, **f_kwargs)
    print(f"Done with GPUs: {resource} - adding them back to the queue")
    q_resources.put(resource)


def run(
    num_folds: int,
    num_folds_used: int,
    num_gpus_total: int,
    num_gpus_per_split: int,
    num_repetitions: int,
    dataset_func: Callable,
    train_func: Callable,
    infer_func: Callable,
    eval_func: Callable,
    dataset_params: Dict = None,
    train_params: Dict = None,
    infer_params: Dict = None,
    eval_params: Dict = None,
    sample_ids: Sequence = None,
):
    """
    ML pipeline - run a full ML experiment pipeline which consists of training on multiple
    cross validation folds, validation inference and evaluation, inference and evaluation
    of each fold's model on a held out test set, ensembling the models trained on different folds,
    and evaluation of the ensembled model.
    The whole process can also be repeated multiple times, to evaluate variability due to different
    random seed.
    Multiple cross validation folds run simultaneously on available GPU resources.

    :param num_folds: Number of cross validation splits/folds.
    :param num_folds_used: Number of folds/splits to use.
        For example, for training a single model with 80% of the samples used
        for training and 20% for validation, set `num_folds=5` and `num_folds_used=1`.
    :param num_gpus_total: Number of GPUs to use in total for executing the pipeline.
    :param num_gpus_per_split: Number of GPUs to use for a single model training/inference.
    :param num_repetitions: Number of repetitions of the procedure with different random
        seeds. Note that this does not change the random decision on cross validation
        fold sample ids.
    :param dataset_func: Callable to a custom function that implements a dataset
        creation. Its input is a path to cache directory and it should return a
        train and test dataset.
    :param train_func: Callable to a custom function that executes model training.
    :param infer_func: Callable to a custom inference function.
    :param eval_func: Callable to a custom evaluation function.
    :param dataset_params: Dictionary that can contain any number of additional
        custom parameters for dataset_func.
    :param train_params: Dictionary that can contain any number of additional
        custom parameters for train_func.
    :param infer_params: Dictionary that can contain any number of additional
        custom parameters for infer_func.
    :param eval_params: Dictionary that can contain any number of additional
        custom parameters for eval_func.
    :param sample_ids: May contain a sequence of array pairs denoting sample ids
        for pre-defined train/validation splits. If this parameter is set to `None`,
        the splits are decided at random.
    """

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for pytorch deterministic mode
    multiprocessing.set_start_method("spawn")
    if num_gpus_total == 0 or num_gpus_per_split == 0:
        if train_params is not None and "manager.train_params" in train_params:
            train_params["manager.train_params"]["device"] = "cpu"

    # set debug mode:
    mode = "default"  # Options: 'default', 'debug'. See details in FuseDebug
    _ = FuseDebug(mode)

    available_gpu_ids = FuseUtilsGPU.get_available_gpu_ids()
    if num_gpus_total < len(available_gpu_ids):
        available_gpu_ids = available_gpu_ids[0:num_gpus_total]
    # group gpus into chunks of size params['common']['num_gpus_per_split']
    gpu_resources = [
        available_gpu_ids[i : i + num_gpus_per_split] for i in range(0, len(available_gpu_ids), num_gpus_per_split)
    ]

    # create a queue of gpu chunks (resources)
    q_resources = Queue()
    for r in gpu_resources:
        q_resources.put(r)

    dataset, test_dataset = dataset_func(**dataset_params)
    if sample_ids is None:
        # the split decision should be the same regardless of repetition index
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1234)
        sample_ids = [item for item in kfold.split(dataset)]
    else:
        assert num_folds == len(sample_ids)

    for rep_index in range(num_repetitions):
        # run training, inference and evaluation on all cross validation folds in parallel
        # using the available gpu resources:
        runner = partial(runner_wrapper, q_resources, rep_index, [train_func, infer_func, eval_func])

        # create process per fold
        processes = [
            Process(target=runner, args=(dataset, ids, cv_index, False, [train_params, infer_params, eval_params]))
            for (ids, cv_index) in zip(sample_ids, range(num_folds))
        ][0:num_folds_used]

        for p in processes:
            p.start()

        for p in processes:
            p.join()
            p.close()

        # infer and eval each split's model on test set:
        runner = partial(runner_wrapper, q_resources, rep_index, [infer_func, eval_func])
        # create process per fold
        processes = [
            Process(target=runner, args=(test_dataset, None, cv_index, True, [infer_params, eval_params]))
            for cv_index in range(num_folds)
        ][0:num_folds_used]
        for p in processes:
            p.start()

        for p in processes:
            p.join()
            p.close()

        # generate ensembled predictions:
        test_dirs = [
            os.path.join(infer_params["paths"]["test_dir"], "rep_" + str(rep_index), str(cv_index))
            for cv_index in range(num_folds)
        ][0:num_folds_used]
        test_infer_filename = infer_params["test_infer_filename"]
        ensembled_output_file = os.path.join(
            infer_params["paths"]["test_dir"], "rep_" + str(rep_index), "ensemble", "ensemble_results.gz"
        )
        ensemble(test_dirs, test_infer_filename, ensembled_output_file)

        # evaluate ensemble:
        rand_gen = Seed.set_seed(rep_index, deterministic_mode=True)
        eval_func(
            dataset=None,
            sample_ids=None,
            cv_index="ensemble",
            test=True,
            params=infer_params,
            rep_index=rep_index,
            rand_gen=rand_gen,
            pred_key="preds",
            label_key="target",
        )
