from fuse.utils.utils_gpu import FuseUtilsGPU
from fuse.utils.utils_debug import FuseUtilsDebug
from sklearn.model_selection import KFold
import multiprocessing
from functools import partial
from multiprocessing import Process, Queue

def setup_dbg():
    ##########################################
    # Debug modes
    ##########################################
    mode = 'default'  # Options: 'default', 'fast', 'debug', 'verbose', 'user'. See details in FuseUtilsDebug
    debug = FuseUtilsDebug(mode)


def runner_wrapper(q_resources, f, *f_args, **f_kwargs):

    resource = q_resources.get()
    print(f"Using GPUs: {resource}")
    FuseUtilsGPU.choose_and_enable_multiple_gpus(len(resource), force_gpus=list(resource))
    f(*f_args, **f_kwargs)
    print(f"Done with GPUs: {resource} - adding them back to the queue")
    q_resources.put(resource)

def run(num_folds, num_gpus_total, num_gpus_per_split, dataset_func, \
        train_func, infer_func, eval_func, \
        dataset_params=None, train_params=None, infer_params=None, eval_params=None):
    #if num_gpus_total == 0 or num_gpus_per_split == 0:
        #kwargs['common']['device'] = 'cpu'
        #params['manager.train_params']['device'] = 'cpu'
    setup_dbg()
    available_gpu_ids = FuseUtilsGPU.get_available_gpu_ids()
    # group gpus into chunks of size params['common']['num_gpus_per_split']
    gpu_resources = [available_gpu_ids[i:i+num_gpus_per_split] for i in range(0, len(available_gpu_ids), num_gpus_per_split)]
    if num_gpus_total < len(available_gpu_ids):
        available_gpu_ids = available_gpu_ids[0:num_gpus_total]
    dataset, test_dataset = dataset_func(**dataset_params)
    n_splits = num_folds if num_folds is not None else 5
    kfold = KFold(n_splits=n_splits, shuffle=True)
    sample_ids = [item for item in kfold.split(dataset)]

    # create a queue of gpu chunks (resources)
    q_resources = Queue()
    for r in gpu_resources:
        q_resources.put(r)

    runner = partial(runner_wrapper, q_resources, train_func)
    # create process per fold
    processes = [Process(target=runner, args=(train_params, dataset, ids, cv_index)) for (ids, cv_index) in zip(sample_ids, range(n_splits))] 
    for p in processes:
        p.start()

    for p in processes:
        p.join()
        p.close()

    #params['infer']['run_func'](params=params)
    #params['eval']['run_func'](params=params)
