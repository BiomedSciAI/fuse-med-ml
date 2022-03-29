import multiprocessing
import time
# can also be a dictionary
gpu_id_list = [1,2,3,4]


def function(x):
    cpu_name = multiprocessing.current_process().name
    cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
    gpu_id = gpu_id_list[cpu_id]
    time.sleep(gpu_id)
    return x * gpu_id


if __name__ == '__main__':
    pool = multiprocessing.Pool(4)
    input_list = [1 for i in range(1)]
    print(pool.map(function, input_list))