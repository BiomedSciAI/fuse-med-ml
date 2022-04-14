import os
import unittest
from testbook import testbook
import fuse.utils.gpu as FuseUtilsGPU

class NotebookHelloWorldTestCase(unittest.TestCase):
    
    def test_notebook(self):
        # Execute the whole notebook and save it as an object
        with testbook('fuse_examples/tutorials/hello_world/hello_world.ipynb', execute=True, timeout=600) as tb:

            test_result_acc = tb.ref("test_result_acc")
            assert(test_result_acc > 0.95)


if __name__ == '__main__':
    unittest.main()