from fuse.dl.cross_validation.pipeline import run
from examples.fuse_examples.imaging.classification.stoic21.runner_stoic21 import run_train, run_infer, run_eval
from examples.fuse_examples.imaging.classification.stoic21.dataset import create_dataset, train_val_test_splits
from examples.fuse_examples.imaging.classification.stoic21.runner_stoic21 import (
    TRAIN_COMMON_PARAMS,
    INFER_COMMON_PARAMS,
    EVAL_COMMON_PARAMS,
    DATASET_COMMON_PARAMS,
)
import os

##########################################
# Required Parameters
##########################################
num_repetitions = 1
num_gpus_total = 3
num_gpus_per_split = 1
num_folds = 4
num_folds_used = 4
dataset_func = create_dataset
train_func = run_train
infer_func = run_infer
eval_func = run_eval
deterministic_mode = True

# output paths:
root_path = "results"
paths = {
    "data_dir": os.environ["STOIC21_DATA_PATH"],
    "cache_dir": os.path.join(root_path, "stoic21/cache_dir"),
    "model_dir": os.path.join(root_path, "stoic21/model_dir"),
    "data_split_filename": os.path.join(root_path, "stoic21/stoic21_split.pkl"),
    "inference_dir": os.path.join(root_path, "stoic21/infer_dir"),
    "eval_dir": os.path.join(root_path, "stoic21/eval_dir"),
}


##########################################
# Custom Parameters
##########################################


##########################################
# Train Params
##########################################
train_params = TRAIN_COMMON_PARAMS
train_params["train_func"] = run_train
train_params["trainer.num_devices"] = num_gpus_per_split
train_params["trainer.strategy"] = None
train_params["trainer.auto_select_gpus"] = False
######################################
# Inference Params
######################################
infer_params = INFER_COMMON_PARAMS
infer_params["run_func"] = run_infer
infer_params["infer_filename"] = "infer.gz"
infer_params["trainer.auto_select_gpus"] = False
infer_params["pred_key"] = "model.output.classification"

######################################
# Eval Params
######################################
eval_params = EVAL_COMMON_PARAMS
eval_params["infer_filename"] = infer_params["infer_filename"]
eval_params["run_func"] = run_eval

##########################################
# Dataset Params
##########################################

dataset_params = DATASET_COMMON_PARAMS
dataset_params["train"] = train_params
dataset_params["infer"] = infer_params
dataset_params["target_key"] = "data.gt.probSevere"

splits = train_val_test_splits(paths=paths, params=dataset_params)
sample_ids_per_fold = [(s[0], s[1]) for s in splits]

if __name__ == "__main__":
    run(
        num_folds,
        num_folds_used,
        num_gpus_total,
        num_gpus_per_split,
        num_repetitions,
        dataset_func,
        train_func,
        infer_func,
        eval_func,
        dataset_params,
        train_params,
        infer_params,
        eval_params,
        paths,
        deterministic_mode,
        sample_ids_per_fold,
    )
