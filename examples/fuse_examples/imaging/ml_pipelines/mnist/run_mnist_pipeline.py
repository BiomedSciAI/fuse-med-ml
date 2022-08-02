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

train_params["trainer.num_epochs"] = 2
train_params["trainer.accelerator"] = "gpu"
# use "dp" strategy temp when working with multiple GPUS - workaround for pytorch lightning issue: https://github.com/Lightning-AI/lightning/issues/11807
train_params["trainer.strategy"] = "dp" if num_gpus_per_split > 1 else None
train_params["trainer.num_devices"] = num_gpus_per_split
train_params["trainer.ckpt_path"] = None # checkpoint to continue from


######################################
# Inference Params
######################################
infer_params = {}
infer_params['paths'] = paths
infer_params['infer_filename'] = 'validation_set_infer.gz'
infer_params['test_infer_filename'] = 'test_set_infer.gz'
infer_params['checkpoint_filename'] = "best_epoch.ckpt"
infer_params['checkpoint'] = 'best'  # Fuse TIP: possible values are 'best', 'last' or epoch_index.
infer_params["trainer.num_devices"] = 1
infer_params["trainer.accelerator"] = "gpu"
infer_params["trainer.strategy"] = None
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