import os
import unittest
from testbook import testbook
import fuse.utils.gpu as FuseUtilsGPU

class NotebookHelloWorldTestCase(unittest.TestCase):

    @unittest.skip("TEMP SKIP") # Test is ready-to-use. Waiting for GPU issue to be resolved.
    def test_notebook(self):
        NUM_OF_CELLS = 36
        notebook_path = "fuse_examples/tutorials/hello_world/hello_world.ipynb"

        # Execute the whole notebook and save it as an object
        with testbook(notebook_path, execute=True, timeout=600) as tb:

            # Sanity check
            test_result_acc = tb.ref("test_result_acc")
            assert(test_result_acc > 0.95)

            # Check that all the notebook's cell executed
            last_cell_output = tb.cell_output_text(NUM_OF_CELLS - 1)
            assert(last_cell_output == 'Done!')


if __name__ == '__main__':
    unittest.main() 