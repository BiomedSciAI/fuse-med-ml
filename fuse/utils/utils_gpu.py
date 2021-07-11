"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

import logging
import os
import subprocess
import traceback
from typing import Any, List, Optional

import torch

from fuse.utils.utils_debug import FuseUtilsDebug


class FuseUtilsGPU():
    @classmethod
    def get_available_gpu_ids(cls) -> List[int]:
        nvidia_smi_output = run_nvidia_smi()
        if nvidia_smi_output is None:
            return None

        available_gpu_ids = get_available_gpu_ids_from_nvidia_smi_output(nvidia_smi_output)
        return available_gpu_ids

    @classmethod
    def set_cuda_visible_devices(cls, list_of_gpu_ids: List[int]) -> None:
        devices = [str(x) for x in list_of_gpu_ids]
        allow_visible_gpus_str = ','.join(devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = allow_visible_gpus_str

    @classmethod
    def choose_and_enable_multiple_gpus(cls, num_gpus_needed: int, force_gpus: Optional[List[int]] = None, use_cpu_if_fail: bool = False) -> int:
        """
        Look for free gpus and set CUDA_VISIBLE_DEVICES accordingly
        :param num_gpus_needed: required number of gpus
        :param force_gpus: Optional, use this list instead of looking for free ones,
        :return: number of gpus allocated
        """
        # debug - num gpus
        try:
            override_num_gpus = FuseUtilsDebug().get_setting('manager_override_num_gpus')
            if override_num_gpus != 'default':
                num_gpus_needed = min(override_num_gpus, num_gpus_needed)
                logging.getLogger('Fuse').info(f'Manager - debug mode - override num_gpus to {num_gpus_needed}', {'color': 'red'})

            if num_gpus_needed == 0:
                return

            if force_gpus is None:
                available_gpu_ids = cls.get_available_gpu_ids()
            else:
                available_gpu_ids = force_gpus

            if available_gpu_ids is None:
                raise Exception('FuseUtilsGPU: could not auto-detect available GPUs')
            elif len(available_gpu_ids) < num_gpus_needed:
                raise Exception('FuseUtilsGPU: not enough GPUs available, requested %d GPUs but only IDs %s are available!' % (
                    num_gpus_needed, str(available_gpu_ids)))
            else:
                selected_gpu_ids = sorted(available_gpu_ids, reverse=True)[:num_gpus_needed]
                logging.getLogger('Fuse').info('FuseUtilsGPU: selecting GPUs %s' % str(selected_gpu_ids))
                cls.set_cuda_visible_devices(selected_gpu_ids)

            torch.backends.cudnn.benchmark = False  # to prevent gpu illegal instruction exceptions
            torch.multiprocessing.set_sharing_strategy('file_system')  # to prevent a bug of too many open file descriptors
        except:
            if use_cpu_if_fail:
                lgr = logging.getLogger('Fuse')
                track = traceback.format_exc()
                lgr.warning(e)
                lgr.warning(track)
                return 0
            else:
                raise

        return num_gpus_needed
    @classmethod
    def move_tensor_to_device(cls, tensor: Any, device: str) -> Any:
        """
        Moves tensor to device (if the type of tensor is torch.Tensor).
        Returns the tensor back.
        :param tensor: data to move to device
        :param device: the device
        :return: tensor
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(device)
        return tensor

    @classmethod
    def allocate_gpu_for_process(cls, gpu_list: list):
        """
        Allocate a gpu for a process given list of available gpus shared between processes
        :param gpu_list: available gpus shared between processes
        :return: None
        """
        try:
            # extract gpu_id dedicated to that process only
            gpu_id = gpu_list.pop()
            if gpu_id == 'cpu':
                gpu_id = ''
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        except:
            lgr = logging.getLogger('Fuse')
            # don't throw error in init function - simply print the error
            track = traceback.format_exc()
            lgr.error(track)

    @classmethod
    def deallocate_gpu(cls):
        import gc
        with torch.no_grad():
            torch.cuda.empty_cache()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.is_cuda:
                        del obj
            except Exception:
                pass
        gc.collect()


def run_nvidia_smi() -> str:
    process = subprocess.Popen("nvidia-smi", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    nvidia_smi_output, stderr = process.communicate()
    status = process.poll()
    if status != 0:
        print("FuseUtilsGPU: Failed to run nvidia-smi")
        return None
    nvidia_smi_output = str(nvidia_smi_output)
    return nvidia_smi_output


def get_available_gpu_ids_from_nvidia_smi_output(raw_output: str) -> List[int]:
    """
    Parse 'nvidia-smi' table
    :param raw_output:
    :return:
    """
    split_output = raw_output.split('\\n')
    current_gpu = -1
    available_gpu_ids = []
    for line_idx, line in enumerate(split_output):
        if 'Off' in line:
            current_gpu = int(split_output[line_idx][4])
            continue
        if current_gpu is not -1:
            loc = line.find('MiB /')
            if loc is not -1:
                memory_usage = int(line[int(loc - 7):loc])
                if memory_usage < 50:
                    available_gpu_ids.append(current_gpu)
                    current_gpu = -1

    return available_gpu_ids


if __name__ == '__main__':
    print(FuseUtilsGPU().get_available_gpu_ids())
    # FuseUtilsGPU().choose_and_enable_multiple_gpus(2)
