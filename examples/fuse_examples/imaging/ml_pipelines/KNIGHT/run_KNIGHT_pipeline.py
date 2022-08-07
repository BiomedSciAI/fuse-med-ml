from fuse.managers.pipeline import run
from funcs import run_train, run_infer, run_eval
from dataset import knight_dataset
import os
import pandas as pd

##########################################
# Required Parameters
##########################################
num_repetitions = 1
num_gpus_total = 3
num_gpus_per_split = 1
num_folds = 5
num_folds_used = 5
dataset_func = knight_dataset
train_func = run_train
infer_func = run_infer
eval_func = run_eval

##########################################
# Custom Parameters
##########################################

# output paths:
root_path = 'results' 
paths = {'data_dir': os.environ['KNIGHT_DATA'], 
         'model_dir': os.path.join(os.environ['KNIGHT_RESULTS'], 'model'),
         'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'cache_dir': os.environ['KNIGHT_CACHE'],
         'inference_dir': os.path.join(os.environ['KNIGHT_RESULTS'], 'infer'),
         'eval_dir': os.path.join(os.environ['KNIGHT_RESULTS'], 'eval'),
         'test_dir': os.path.join(os.environ['KNIGHT_RESULTS'], 'test'), 
         }

# common params:
common_params = {}
common_params['task_num'] = 1
common_params['use_data'] = {'imaging': False, 'clinical': True} # specify whether to use imaging, clinical data or both
common_params['target_name'] = 'data.gt.gt_global.task_1_label'
common_params['target_metric'] = 'validation.metrics.auc'
common_params['num_classes'] = 2
common_params['num_gpus_per_split'] = num_gpus_per_split


##########################################
# Dataset Params
##########################################

dataset_params = {}
dataset_params['data_dir'] = paths['data_dir']
dataset_params['cache_dir'] = paths['cache_dir']
dataset_params['resize_to'] = (256, 256, 110) 
dataset_params['reset_cache'] = False
# custom train/test split:
splits_path = '../../classification/knight/baseline/splits_final.pkl'
splits=pd.read_pickle(splits_path)
split = splits[0] 
# note that the validation set of this split will be used as *test* for the ML pipeline example
# the cross validation splits will be drawn randomly from split['train'] 
dataset_params['split'] = split

##########################################
# Train Params
##########################################
train_params = {}
train_params['dataset'] = dataset_params
train_params['split'] = split
train_params['paths'] = paths
train_params['common'] = common_params
train_params['imaging_dropout'] = 0.5
train_params['clinical_dropout'] = 0.0
train_params['fused_dropout'] = 0.5
train_params['batch_size'] = 2
train_params['num_epochs'] = 2
train_params['learning_rate'] = 1e-4
train_params['num_workers'] = 8


######################################
# Inference Params
######################################
infer_params = {}
infer_params['dataset'] = dataset_params
infer_params['train'] = train_params
infer_params['paths'] = paths
infer_params['common'] = common_params
infer_params['test_infer_filename'] = 'test_results.gz'
infer_params['checkpoint'] = 'best'  # Fuse TIP: possible values are 'best', 'last' or epoch_index.

# ===============
# Run function
# ===============
infer_params['run_func'] = run_infer

######################################
# Eval Params
######################################
eval_params = {}
eval_params['paths'] = paths
eval_params['common'] = common_params
eval_params['test_infer_filename'] = infer_params['test_infer_filename']
eval_params['run_func'] = run_eval



if __name__ == "__main__":
    run(num_folds, num_folds_used, num_gpus_total, num_gpus_per_split, \
        num_repetitions, dataset_func, train_func, infer_func, eval_func, \
        dataset_params, train_params, infer_params, eval_params)