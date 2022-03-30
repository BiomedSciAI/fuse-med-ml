from fuse.managers.pipeline import run
#from fuse_examples.classification.mnist.runner import run_train, run_infer, run_eval
from funcs import run_train, run_infer, run_eval, create_dataset
import os

params = {}

##########################################
# Output Paths
##########################################
root_path = 'examples' 
paths = {'model_dir': os.path.join(root_path, 'mnist/model_dir'),
         'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'cache_dir': os.path.join(root_path, 'mnist/cache_dir'),
         'inference_dir': os.path.join(root_path, 'mnist/infer_dir'),
         'eval_dir': os.path.join(root_path, 'mnist/eval_dir')}

##########################################
# Common Params
##########################################
common_params = {}
common_params['paths'] = paths
common_params['num_gpus'] = 3
common_params['num_folds'] = 5

##########################################
# Train Params
##########################################
train_params = {}
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
# Manager - Train
# ===============
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
train_params['manager.learning_rate'] = 1e-4
train_params['manager.weight_decay'] = 0.001
train_params['manager.resume_checkpoint_filename'] = None  # if not None, will try to load the checkpoint

# ===============
# Dataset func.
# ===============
train_params['dataset_func'] = create_dataset

# ===============
# Run func.
# ===============
train_params['run_func'] = run_train

######################################
# Inference Params
######################################
infer_params = {}
infer_params['infer_filename'] = 'validation_set_infer.gz'
infer_params['checkpoint'] = 'best'  # Fuse TIP: possible values are 'best', 'last' or epoch_index.

# ===============
# Run function
# ===============
infer_params['run_func'] = run_infer

######################################
# Eval Params
######################################
eval_params = {}
eval_params['infer_filename'] = infer_params['infer_filename']
eval_params['run_func'] = run_eval

params['common'] = common_params
params['train'] = train_params
params['infer'] = infer_params
params['eval'] = eval_params



if __name__ == "__main__":
    run(params=params)