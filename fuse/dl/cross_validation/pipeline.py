from typing import Dict, Callable
from fuse.utils import gpu as FuseUtilsGPU
from sklearn.model_selection import KFold
from functools import partial
from multiprocessing import Process, Queue
from typing import Sequence, Union
import os
from fuse.eval.metrics.classification.metrics_ensembling_common import MetricEnsemble
from collections import OrderedDict
from fuse.eval.evaluator import EvaluatorDefault
from fuse.utils.file_io.file_io import create_or_reset_dir
from fuse.utils.rand.seed import Seed
from multiprocessing import set_start_method

set_start_method("spawn", force=True)


def ensemble(
    test_dirs: Sequence[str], test_infer_filename: str, pred_key: str, target_key: str, ensembled_output_file: str
) -> None:
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
                    pred_keys=[str(i) + "." + pred_key for i in range(len(test_dirs))],
                    target="0." + target_key,
                    output_file=ensembled_output_file,
                    output_pred_key=pred_key,
                    output_target_key=target_key,
                ),
            ),
        ]
    )

    evaluator = EvaluatorDefault()
    _ = evaluator.eval(ids=None, data=data, metrics=metrics)


def runner_wrapper(
    q_resources: Queue,
    rep_index: int,
    deterministic_mode: bool,
    fs: Union[Sequence[Callable], Callable],
    *f_args: tuple,
    **f_kwargs: dict,
) -> None:
    resource = q_resources.get()
    print(f"Using GPUs: {resource}")
    FuseUtilsGPU.choose_and_enable_multiple_gpus(len(resource), force_gpus=list(resource))
    _ = Seed.set_seed(rep_index, deterministic_mode=deterministic_mode)
    if isinstance(fs, Sequence):
        for f, prev_arg, last_arg in zip(fs, f_args[-2], f_args[-1]):
            f(*(f_args[:-2] + (prev_arg,) + (last_arg,)), **f_kwargs)
    else:
        f(*f_args, **f_kwargs)
    print(f"Done with GPUs: {resource} - adding them back to the queue")
    q_resources.put(resource)


def train_wrapper(
    sample_ids_per_fold: Sequence,
    cv_index: int,
    rep_index: int,
    dataset_func: Callable,
    dataset_params: dict,
    paths: dict,
    func: Callable,
    params: dict,
) -> None:

    paths_train = paths.copy()

    # set parameters specific to this fold:
    paths_train["model_dir"] = os.path.join(paths["model_dir"], "rep_" + str(rep_index), str(cv_index))
    paths_train["cache_dir"] = os.path.join(paths["cache_dir"], "rep_" + str(rep_index), str(cv_index))

    # generate data for this fold:
    train_dataset, validation_dataset = dataset_func(
        train_val_sample_ids=sample_ids_per_fold, paths=paths_train, params=dataset_params
    )

    # call project specific train_func:
    func(train_dataset=train_dataset, validation_dataset=validation_dataset, paths=paths_train, train_params=params)


def infer_wrapper(
    sample_ids_per_fold: Sequence,
    cv_index: int,
    rep_index: int,
    dataset_func: Callable,
    dataset_params: dict,
    paths: dict,
    func: Callable,
    params: dict,
) -> None:

    paths_infer = paths.copy()

    # set parameters specific to this fold, and generate data:
    paths_infer["model_dir"] = os.path.join(paths["model_dir"], "rep_" + str(rep_index), str(cv_index))
    if sample_ids_per_fold is None:  # test mode
        paths_infer["inference_dir"] = os.path.join(
            paths["inference_dir"], "test", "rep_" + str(rep_index), str(cv_index)
        )
        _, dataset = dataset_func(train_val_sample_ids=None, paths=paths_infer, params=dataset_params)
    else:
        paths_infer["inference_dir"] = os.path.join(
            paths["inference_dir"], "validation", "rep_" + str(rep_index), str(cv_index)
        )
        _, dataset = dataset_func(train_val_sample_ids=sample_ids_per_fold, paths=paths_infer, params=dataset_params)

    # call project specific infer_func:
    func(dataset=dataset, paths=paths_infer, infer_params=params)


def eval_wrapper(
    sample_ids_per_fold: Sequence,
    cv_index: int,
    rep_index: int,
    dataset_func: Callable,
    dataset_params: dict,
    paths: dict,
    func: Callable,
    params: dict,
) -> None:

    paths_eval = paths.copy()

    if sample_ids_per_fold is None:  # test mode
        paths_eval["inference_dir"] = os.path.join(
            paths["inference_dir"], "test", "rep_" + str(rep_index), str(cv_index)
        )
        paths_eval["eval_dir"] = os.path.join(paths["eval_dir"], "test", "rep_" + str(rep_index), str(cv_index))
    else:
        paths_eval["inference_dir"] = os.path.join(
            paths["inference_dir"], "validation", "rep_" + str(rep_index), str(cv_index)
        )
        paths_eval["eval_dir"] = os.path.join(paths["eval_dir"], "validation", "rep_" + str(rep_index), str(cv_index))

    # call project specific eval_func:
    func(paths=paths_eval, eval_params=params)


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
    paths: Dict = None,
    deterministic_mode: bool = True,
    sample_ids_per_fold: Sequence = None,
) -> None:
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
        For running a 5-fold cross-validation scenario, set `num_folds=5` and `num_folds_used=5`.
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
    :param deterministic_mode: Whether to use PyTotch deterministic mode.
    :param sample_ids_per_fold: May contain a sequence of array pairs denoting sample ids
        for pre-defined train/validation splits. If this parameter is set to `None`,
        the splits are decided at random.
    """
    available_gpu_ids = FuseUtilsGPU.get_available_gpu_ids()
    if num_gpus_total < len(available_gpu_ids):
        available_gpu_ids = available_gpu_ids[0:num_gpus_total]
    # group gpus into chunks of size num_gpus_per_split
    gpu_resources = [
        available_gpu_ids[i : i + num_gpus_per_split] for i in range(0, len(available_gpu_ids), num_gpus_per_split)
    ]

    # create a queue of gpu chunks (resources)
    q_resources = Queue()
    for r in gpu_resources:
        q_resources.put(r)

    if sample_ids_per_fold is None:
        dataset, test_dataset = dataset_func(paths=paths, params=dataset_params)
        # the split decision should be the same regardless of repetition index
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1234)
        sample_ids_per_fold = [item for item in kfold.split(dataset)]
    else:
        assert num_folds == len(sample_ids_per_fold)

    for rep_index in range(num_repetitions):
        # run training, inference and evaluation on all cross validation folds in parallel
        # using the available gpu resources:
        runner = partial(
            runner_wrapper, q_resources, rep_index, deterministic_mode, [train_wrapper, infer_wrapper, eval_wrapper]
        )

        # create process per fold
        processes = [
            Process(
                target=runner,
                args=(
                    ids,
                    cv_index,
                    rep_index,
                    dataset_func,
                    dataset_params,
                    paths,
                    [train_func, infer_func, eval_func],
                    [train_params, infer_params, eval_params],
                ),
            )
            for (ids, cv_index) in zip(sample_ids_per_fold, range(num_folds))
        ][0:num_folds_used]
        for p in processes:
            p.start()

        for p in processes:
            p.join()
            p.close()

        # infer and eval each split's model on test set:
        runner = partial(runner_wrapper, q_resources, rep_index, deterministic_mode, [infer_wrapper, eval_wrapper])
        # create process per fold
        processes = [
            Process(
                target=runner,
                args=(
                    None,
                    cv_index,
                    rep_index,
                    dataset_func,
                    dataset_params,
                    paths,
                    [infer_func, eval_func],
                    [infer_params, eval_params],
                ),
            )
            for cv_index in range(num_folds)
        ][0:num_folds_used]
        for p in processes:
            p.start()

        for p in processes:
            p.join()
            p.close()

        # generate ensembled predictions:
        test_dirs = [
            os.path.join(paths["inference_dir"], "test", "rep_" + str(rep_index), str(cv_index))
            for cv_index in range(num_folds)
        ][0:num_folds_used]
        test_infer_filename = "infer.gz"
        ensembled_output_file = os.path.join(
            paths["inference_dir"], "test", "rep_" + str(rep_index), "ensemble", "infer.gz"
        )

        ensemble(
            test_dirs,
            test_infer_filename,
            infer_params["pred_key"],
            dataset_params["target_key"],
            ensembled_output_file,
        )

        # evaluate ensemble:
        paths_eval = paths.copy()
        paths_eval["inference_dir"] = os.path.join(paths["inference_dir"], "test", "rep_" + str(rep_index), "ensemble")
        paths_eval["eval_dir"] = os.path.join(paths["eval_dir"], "test", "rep_" + str(rep_index), "ensemble")

        _ = Seed.set_seed(rep_index, deterministic_mode=deterministic_mode)
        eval_func(paths=paths_eval, eval_params=eval_params)
