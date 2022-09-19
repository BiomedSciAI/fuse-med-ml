from fuse.dl.cross_validation.pipeline import run
from examples.fuse_examples.imaging.classification.mnist.run_mnist import (
    run_train,
    run_infer,
    run_eval,
    TRAIN_COMMON_PARAMS,
    INFER_COMMON_PARAMS,
    EVAL_COMMON_PARAMS,
)
from funcs import create_dataset
import os

##########################################
# Required Parameters
##########################################
num_repetitions = 1
num_gpus_total = 3
num_gpus_per_split = 1
num_folds = 5
num_folds_used = 5
dataset_func = create_dataset
train_func = run_train
infer_func = run_infer
eval_func = run_eval
deterministic_mode = True

# output paths:
root_path = "results"
paths = {
    "cache_dir": os.path.join(root_path, "mnist/cache_dir"),
    "model_dir": os.path.join(root_path, "mnist/model_dir"),
    "inference_dir": os.path.join(root_path, "mnist/infer_dir"),
    "eval_dir": os.path.join(root_path, "mnist/eval_dir"),
}

##########################################
# Custom Parameters
##########################################

##########################################
# Dataset Params
##########################################

dataset_params = {}
dataset_params["target_key"] = "data.label"

##########################################
# Train Params
##########################################
train_params = TRAIN_COMMON_PARAMS

# use "dp" strategy temp when working with multiple GPUS - workaround for pytorch lightning issue: https://github.com/Lightning-AI/lightning/issues/11807
train_params["trainer.strategy"] = "dp" if num_gpus_per_split > 1 else None
train_params["trainer.num_devices"] = num_gpus_per_split

######################################
# Inference Params
######################################
infer_params = INFER_COMMON_PARAMS
infer_params["infer_filename"] = "infer.gz"
infer_params["pred_key"] = "model.output.classification"

# ===============
# Run function
# ===============
infer_params["run_func"] = run_infer


######################################
# Eval Params
######################################
eval_params = EVAL_COMMON_PARAMS
eval_params["infer_filename"] = infer_params["infer_filename"]
eval_params["run_func"] = run_eval


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
    )
