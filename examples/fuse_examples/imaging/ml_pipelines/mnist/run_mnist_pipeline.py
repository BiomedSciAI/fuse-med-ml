from fuse.managers.pipeline import run
from funcs import run_train, run_infer, run_eval, create_dataset
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

# output paths:
root_path = 'results' 
paths = {'model_dir': os.path.join(root_path, 'mnist/model_dir'),
         'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'cache_dir': os.path.join(root_path, 'mnist/cache_dir'),
         'inference_dir': os.path.join(root_path, 'mnist/infer_dir'),
         'eval_dir': os.path.join(root_path, 'mnist/eval_dir'),
         'test_dir': os.path.join(root_path, 'mnist/test_dir'), 
         }

##########################################
# Custom Parameters
##########################################

##########################################
# Dataset Params
##########################################

dataset_params = {}
dataset_params['cache_dir'] = paths['cache_dir']

##########################################
# Train Params
##########################################
train_params = {}
train_params['paths'] = paths

# ============
# Model
# ============
train_params['model'] = 'lenet' # 'resnet18' or 'lenet'

# ============
# Data
# ============
if train_params['model'] == 'lenet':
    train_params['data.batch_size'] = 100
elif train_params['model'] == 'resnet18':
    train_params['data.batch_size'] = 30
train_params['data.train_num_workers'] = 8
train_params['data.validation_num_workers'] = 8

# ===============
# Train
# ===============
train_params['opt.learning_rate'] = 1e-4
train_params['opt.weight_decay'] = 0.001

train_params["trainer.num_epochs"] = 5
train_params["trainer.accelerator"] = ""
train_params["trainer.strategy"] = ""


train_params['manager.train_params'] = {
    'device': 'cuda', 
    'num_epochs': 5,
    'virtual_batch_size': 1,  # number of batches in one virtual batch
    'start_saving_epochs': 10,  # first epoch to start saving checkpoints from
    'gap_between_saving_epochs': 5,  # number of epochs between saved checkpoint
}
train_params['manager.best_epoch_source'] = {
    'source': 'metrics.accuracy',  # can be any key from 'epoch_results'
    'optimization': 'max',  # can be either min/max
    'on_equal_values': 'better',
    # can be either better/worse - whether to consider best epoch when values are equal
}
train_params['manager.resume_checkpoint_filename'] = None  # if not None, will try to load the checkpoint


######################################
# Inference Params
######################################
infer_params = {}
infer_params['paths'] = paths
infer_params['infer_filename'] = 'validation_set_infer.gz'
infer_params['test_infer_filename'] = 'test_set_infer.gz'
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
eval_params['infer_filename'] = infer_params['infer_filename']
eval_params['test_infer_filename'] = infer_params['test_infer_filename']
eval_params['run_func'] = run_eval



if __name__ == "__main__":
    run(num_folds, num_folds_used, num_gpus_total, num_gpus_per_split, \
        num_repetitions, dataset_func, train_func, infer_func, eval_func, \
        dataset_params, train_params, infer_params, eval_params)