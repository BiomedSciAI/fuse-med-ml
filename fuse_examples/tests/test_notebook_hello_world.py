import os
import unittest
from testbook import testbook
import fuse.utils.gpu as FuseUtilsGPU

class NotebookHelloWorldTestCase(unittest.TestCase):
    
    def test_notebook(self):
        notebook_path = "fuse_examples/tutorials/hello_world/hello_world.ipynb"
        # Execute the whole notebook and save it as an object
        with testbook(notebook_path, execute=True, timeout=600) as tb:
            # TODO!!!!
            # Fix that it won't install each tim
            test_result_acc = tb.ref("test_result_acc")
            tb.cell_output_text(1)
            assert(test_result_acc > 0.95)


if __name__ == '__main__':
    unittest.main()