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

import shutil
import tempfile
import unittest

import torch

from fuse.utils.multiprocessing.run_multiprocessed import run_in_subprocess
from fuse.utils.rand.seed import Seed
from fuse.utils.tests.decorators import skipIfMultiple
from fuse_examples.imaging.classification.mnist.simple_mnist_starter import run_train
from fuseimg.datasets.mnist import MNIST


def run_mnist(root: str) -> None:
    Seed.set_seed(
        0, False
    )  # previous test (in the pipeline) changed the deterministic behavior to True
    try:
        run_train(num_epochs=1)
    except:
        raise Exception("Training MNIST Failed.")


@skipIfMultiple(
    (not torch.cuda.is_available(), "No GPU is available"),
)
class ClassificationMnistTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()

    def test_template(self) -> None:
        run_in_subprocess(run_mnist, self.root)

    def tearDown(self) -> None:
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
