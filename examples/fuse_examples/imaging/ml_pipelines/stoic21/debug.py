from fuse.utils import gpu as FuseUtilsGPU
import torch
import os
import numpy as np

for i in range(20):
    gpuid = np.random.randint(8)
    #gpu_list = [3,5]
    FuseUtilsGPU.choose_and_enable_multiple_gpus(1, force_gpus=[gpuid])
    actual_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    assert(actual_gpu==gpuid)
    #print(f'number of gpus: {torch.cuda.device_count()}')
