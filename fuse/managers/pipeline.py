from typing import Dict
from fuse.utils import gpu as FuseUtilsGPU
from fuse.utils.utils_debug import FuseUtilsDebug
from sklearn.model_selection import KFold
import multiprocessing
from functools import partial
from multiprocessing import Process, Queue
from typing import Sequence
import numpy as np
import pandas as pd
import os

def ensemble(test_dirs, test_infer_filename, ensembled_output_dir):
    test_infer_filenames = [os.path.join(d, test_infer_filename) for d in test_dirs]
    softmax = [pd.read_pickle(f)['model.output.classification'] for f in test_infer_filenames]
    softmax = np.vstack(softmax)
    softmax = np.mean(softmax, 0) # ensemble
    original_infer_1st_split = pd.read_pickle(test_infer_filenames[0])
    ensembled_infer = original_infer_1st_split
    ensembled_infer['model.output.classification'] = softmax



def runner_wrapper(q_resources, fs, *f_args, **f_kwargs):
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

def run(num_folds, num_gpus_total, num_gpus_per_split, dataset_func, \
        train_func, infer_func, eval_func, \
        dataset_params=None, train_params=None, infer_params=None, \
        eval_params=None, sample_ids=None):
    multiprocessing.set_start_method('spawn') 
    if num_gpus_total == 0 or num_gpus_per_split == 0:
        if train_params is not None and 'manager.train_params' in train_params:
            train_params['manager.train_params']['device'] = 'cpu'
    
    # set debug mode:
    mode = 'default'  # Options: 'default', 'fast', 'debug', 'verbose', 'user'. See details in FuseUtilsDebug
    debug = FuseUtilsDebug(mode)

    available_gpu_ids = FuseUtilsGPU.get_available_gpu_ids()
    if num_gpus_total < len(available_gpu_ids):
        available_gpu_ids = available_gpu_ids[0:num_gpus_total]
    # group gpus into chunks of size params['common']['num_gpus_per_split']
    gpu_resources = [available_gpu_ids[i:i+num_gpus_per_split] for i in range(0, len(available_gpu_ids), num_gpus_per_split)]
    dataset, test_dataset = dataset_func(**dataset_params)
    if sample_ids is None:
        kfold = KFold(n_splits=num_folds, shuffle=True)
        sample_ids = [item for item in kfold.split(dataset)]
    else:
        assert(num_folds == len(sample_ids))

    # create a queue of gpu chunks (resources)
    q_resources = Queue()
    for r in gpu_resources:
        q_resources.put(r)

    # run training, inference and evaluation on all cross validation folds in parallel
    # using the available gpu resources:
    runner = partial(runner_wrapper, q_resources, [train_func, infer_func, eval_func])
    # create process per fold
    processes = [Process(target=runner, args=(dataset, ids, cv_index, False, [train_params, infer_params, eval_params])) for (ids, cv_index) in zip(sample_ids, range(num_folds))] 
    for p in processes:
        p.start()

    for p in processes:
        p.join()
        p.close()

    # infer and eval each split's model on test set:
    runner = partial(runner_wrapper, q_resources, [infer_func, eval_func])
    # create process per fold
    processes = [Process(target=runner, args=(test_dataset, None, cv_index, True, [infer_params, eval_params])) for cv_index in range(num_folds)] 
    for p in processes:
        p.start()

    for p in processes:
        p.join()
        p.close()
    
    
    ensemble(test_dirs, test_infer_filename, ensembled_output_dir)
    

    # infer and eval ensembled model model on test set:


    # run infer
    #run_infer(test_dataset, sample_ids=None, cv_index='ensemble', params=infer_params)
