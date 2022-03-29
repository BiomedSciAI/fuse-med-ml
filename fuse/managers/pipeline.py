from fuse.utils.utils_gpu import FuseUtilsGPU
from fuse.utils.utils_debug import FuseUtilsDebug

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
    FuseUtilsGPU.choose_and_enable_multiple_gpus(num_gpus, force_gpus=force_gpus)

def run(params):
    if params['common']['num_gpus'] == 0:
        params['manager.train_params']['device'] = 'cpu'
    setup_dbg()
    available_gpu_ids = FuseUtilsGPU.get_available_gpu_ids()
    
    dataset = params['train']['dataset_func'](params=params)
    params['train']['run_func'](params=params)
    params['infer']['run_func'](params=params)
    params['eval']['run_func'](params=params)
