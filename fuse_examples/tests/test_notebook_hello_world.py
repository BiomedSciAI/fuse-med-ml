import os
import unittest
from testbook import testbook
import fuse.utils.gpu as FuseUtilsGPU

class NotebookHelloWorldTestCase(unittest.TestCase):
    
    def test_notebook(self):
        notebook_path = "fuse_examples/tutorials/hello_world/hello_world.ipynb"
        # Execute the whole notebook and save it as an object
<<<<<<< HEAD
        with testbook(notebook_path, execute=True, timeout=600) as tb:
            # TODO!!!!
            # Fix that it won't install each tim
=======
        with testbook('fuse_examples/tutorials/hello_world/hello_world.ipynb', execute=range(3), timeout=600) as tb:
            print(tb.cell_output_text(2))
       
        with testbook('fuse_examples/tutorials/hello_world/hello_world.ipynb', execute=True, timeout=600) as tb:
            print(tb.cell_output_text(2))
>>>>>>> 48a717349afd14cd5cebac8cf83c25c499b4ebff
            test_result_acc = tb.ref("test_result_acc")
            tb.cell_output_text(1)
            assert(test_result_acc > 0.95)


if __name__ == '__main__':
    unittest.main()