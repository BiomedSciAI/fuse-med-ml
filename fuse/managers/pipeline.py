from fuse.utils.utils_gpu import FuseUtilsGPU
from fuse.utils.utils_debug import FuseUtilsDebug
from sklearn.model_selection import KFold
import multiprocessing

def setup_dbg():
    ##########################################
    # Debug modes
    ##########################################
    mode = 'default'  # Options: 'default', 'fast', 'debug', 'verbose', 'user'. See details in FuseUtilsDebug
    debug = FuseUtilsDebug(mode)

def choose_gpus(num_gpus, force_gpus=None):
    ##########################################
    # Allocate GPUs
    ##########################################
    # pass specific gpus to force_gpus to force them rather than automatically looking for free ones.
    # for example: force_gpus = [1,4,5] will use these three specifically 
    # choose gpu id for this process
    cpu_name = multiprocessing.current_process().name
    cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
    gpu_id = available_gpu_ids[cpu_id]
    FuseUtilsGPU.choose_and_enable_multiple_gpus(1, force_gpus=[gpu_id])

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

    # create a pool of workers for each available gpu
    pool = multiprocessing.Pool(len(available_gpu_ids), initializer=choose_gpus(num_gpus_per_split))

    pool.starmap(train_func, [(train_params, dataset, available_gpu_ids, ids, cv_index) for (ids, cv_index) in zip(sample_ids, range(n_splits))])
    #params['infer']['run_func'](params=params)
    #params['eval']['run_func'](params=params)
