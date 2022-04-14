import os
import unittest
from testbook import testbook
import fuse.utils.gpu as FuseUtilsGPU

class NotebookHelloWorldTestCase(unittest.TestCase):
    
    def test_notebook(self):
        num_gpus_allocated = FuseUtilsGPU.choose_and_enable_multiple_gpus(1, use_cpu_if_fail=True)
        if num_gpus_allocated == 0:
            assert False, "fail to allocate gpu"

        # Execute the whole notebook and save it as an object
        with testbook('fuse_examples/tutorials/hello_world/hello_world.ipynb', execute=range(3), timeout=600) as tb:
            print(tb.cell_output_text(2))
       
        with testbook('fuse_examples/tutorials/hello_world/hello_world.ipynb', execute=True, timeout=600) as tb:
            print(tb.cell_output_text(2))
            test_result_acc = tb.ref("test_result_acc")
            assert(test_result_acc > 0.95)


if __name__ == '__main__':
    unittest.main()