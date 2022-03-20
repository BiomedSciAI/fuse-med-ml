from fuse.utils.utils_gpu import FuseUtilsGPU
from fuse.utils.utils_debug import FuseUtilsDebug

def setup(num_gpus, force_gpus=None):
    ##########################################
    # Allocate GPUs
    ##########################################
    # pass specific gpus to force_gpus to force them rather than automatically looking for free ones.
    # for example: force_gpus = [1,4,5] will use these three specifically 
    FuseUtilsGPU.choose_and_enable_multiple_gpus(num_gpus, force_gpus=force_gpus)
    
    ##########################################
    # Debug modes
    ##########################################
    mode = 'default'  # Options: 'default', 'fast', 'debug', 'verbose', 'user'. See details in FuseUtilsDebug
    debug = FuseUtilsDebug(mode)
    
def run(params):
    if params['num_gpus'] == 0:
        params['manager.train_params']['device'] = 'cpu'
    setup(num_gpus=params['num_gpus'])
    params['train']['run_func'](paths=params['paths'], train_params=params['train'])
    params['infer']['run_func'](paths=params['paths'], infer_common_params=params['infer'])
    params['eval']['run_func'](paths=params['paths'], eval_common_params=params['eval'])
