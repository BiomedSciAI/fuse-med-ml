import unittest
from testbook import testbook
from fuse.utils.multiprocessing.run_multiprocessed import run_in_subprocess


class NotebookHelloWorldTestCase(unittest.TestCase):
    @run_in_subprocess()
    def test_notebook(self):
        notebook_path = "examples/fuse_examples/imaging/hello_world/hello_world.ipynb"

        # Execute the whole notebook and save it as an object
        with testbook(notebook_path, execute=True, timeout=600) as tb:

            # Sanity check
            test_result_acc = tb.ref("test_result_acc")
            assert test_result_acc > 0.95


if __name__ == "__main__":
    unittest.main()
