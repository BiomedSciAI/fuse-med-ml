import os
import unittest
from testbook import testbook
import fuse.utils.gpu as FuseUtilsGPU

class NotebookHelloWorldTestCase(unittest.TestCase):
    
    def test_notebook(self):
        notebook_path = "fuse_examples/tutorials/hello_world/hello_world.ipynb"
        # Execute the whole notebook and save it as an object

if __name__ == '__main__':
    unittest.main()