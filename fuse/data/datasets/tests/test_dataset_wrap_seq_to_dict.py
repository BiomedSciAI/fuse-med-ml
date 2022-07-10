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

"""

import os
import unittest

import random

import torchvision
from torchvision import transforms
from fuse.utils.rand.seed import Seed
from fuse.utils.ndict import NDict

import tempfile
from fuse.data.datasets.dataset_wrap_seq_to_dict import DatasetWrapSeqToDict


class TestDatasetWrapSeqToDict(unittest.TestCase):
    """
    Test sample caching
    """

    def setUp(self):
        pass

    def test_dataset_wrap_seq_to_dict(self):
        Seed.set_seed(1234)
        path = tempfile.mkdtemp()

        # Create dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        mnist_data_path = os.environ.get("MNIST_DATA_PATH", tempfile.mkdtemp(prefix="mnist"))
        torch_train_dataset = torchvision.datasets.MNIST(
            mnist_data_path, download=True, train=True, transform=transform
        )
        # wrapping torch dataset
        train_dataset = DatasetWrapSeqToDict(
            name="train", dataset=torch_train_dataset, sample_keys=("data.image", "data.label")
        )
        train_dataset.create()

        # get value
        index = random.randint(0, len(train_dataset))
        sample = train_dataset[index]
        item = torch_train_dataset[index]

        self.assertTrue(isinstance(sample, dict))
        self.assertTrue("data.image" in sample)
        self.assertTrue("data.label" in sample)
        self.assertTrue((sample["data.image"] == item[0]).all())
        self.assertEqual(sample["data.label"], item[1])

    def test_dataset_cache(self):
        Seed.set_seed(1234)

        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
        # Create dataset
        mnist_data_path = os.environ.get("MNIST_DATA_PATH", tempfile.mkdtemp(prefix="mnist"))
        torch_dataset = torchvision.datasets.MNIST(mnist_data_path, download=True, train=True, transform=None)
        print(f"torch dataset size = {len(torch_dataset)}")

        # wrapping torch dataset
        cache_dir = tempfile.mkdtemp()

        dataset = DatasetWrapSeqToDict(
            name="test", dataset=torch_dataset, sample_keys=("data.image", "data.label"), cache_dir=cache_dir
        )
        dataset.create()

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
