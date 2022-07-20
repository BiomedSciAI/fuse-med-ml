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

import os
import tempfile
import unittest
import pandas as pds
import numpy as np
from tqdm.std import tqdm
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from fuse.utils import Seed

from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.dataset_wrap_seq_to_dict import DatasetWrapSeqToDict
from fuse.data.utils.collates import CollateDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.utils.samplers import BatchSamplerDefault


class TestSamplers(unittest.TestCase):
    def setUp(self):
        pass

    def test_balanced_dataset(self):
        Seed.set_seed(1234)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # Create dataset
        mnist_data_path = os.environ.get("MNIST_DATA_PATH", tempfile.mkdtemp(prefix="mnist"))
        torch_dataset = torchvision.datasets.MNIST(mnist_data_path, download=True, train=True, transform=transform)
        print(f"torch dataset size = {len(torch_dataset)}")

        num_classes = 10
        num_samples = len(torch_dataset)

        # wrapping torch dataset
        dataset = DatasetWrapSeqToDict(name="test", dataset=torch_dataset, sample_keys=("data.image", "data.label"))
        dataset.create()
        print(dataset.summary())
        batch_sampler = BatchSamplerDefault(
            dataset=dataset,
            balanced_class_name="data.label",
            num_balanced_classes=num_classes,
            batch_size=32,
            mode="approx",
            balanced_class_weights=[1 / num_classes] * num_classes,
            workers=10,
        )

        labels = np.zeros(num_classes)

        # Create dataloader
        dataloader = DataLoader(
            dataset=dataset, collate_fn=CollateDefault(), batch_sampler=batch_sampler, shuffle=False, drop_last=False
        )
        iter1 = iter(dataloader)
        for _ in tqdm(range(len(dataloader))):
            batch_dict = next(iter1)
            labels_in_batch = batch_dict["data.label"]
            for label in labels_in_batch:
                labels[label] += 1

        # final balance
        print(labels)
        for idx in range(num_classes):
            sampled = labels[idx] / num_samples
            print(f"Class {idx}: {sampled * 100}% of data")
            self.assertAlmostEqual(
                sampled,
                1 / num_classes,
                delta=1 / num_classes * 0.5,
                msg=f"Unbalanced class {idx}, expected 0.1+-0.05 and got {sampled}",
            )

    def test_not_equalbalance_dataset(self):
        Seed.set_seed(1234)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # Create dataset
        mnist_data_path = os.environ.get("MNIST_DATA_PATH", tempfile.mkdtemp(prefix="mnist"))
        torch_dataset = torchvision.datasets.MNIST(mnist_data_path, download=True, train=True, transform=transform)
        print(f"torch dataset size = {len(torch_dataset)}")

        num_classes = 10
        probs = 1 / num_classes

        # wrapping torch dataset
        dataset = DatasetWrapSeqToDict(name="test", dataset=torch_dataset, sample_keys=("data.image", "data.label"))
        dataset.create()

        balanced_class_weights = [1] * 5 + [3] * 5
        batch_size = 20
        batch_sampler = BatchSamplerDefault(
            dataset=dataset,
            balanced_class_name="data.label",
            num_balanced_classes=num_classes,
            batch_size=batch_size,
            mode="exact",
            balanced_class_weights=balanced_class_weights,
        )

        # Create dataloader
        labels = np.zeros(num_classes)
        dataloader = DataLoader(
            dataset=dataset, collate_fn=CollateDefault(), batch_sampler=batch_sampler, shuffle=False, drop_last=False
        )
        iter1 = iter(dataloader)
        num_items = 0
        for _ in tqdm(range(len(dataloader))):
            batch_dict = next(iter1)
            labels_in_batch = batch_dict["data.label"]
            for label in labels_in_batch:
                labels[label] += 1
                num_items += 1

        # final balance
        print(labels)
        for idx in range(num_classes):
            sampled = labels[idx] / num_items
            print(f"Class {idx}: {sampled * 100}% of data")
            self.assertEqual(sampled, balanced_class_weights[idx] / batch_size)

    def test_sampler_default(self):
        # datainfo
        data = {
            "sample_id": ["a", "b", "c", "d", "e"],
            "data.values": [7, 4, 9, 2, 4],
            "data.class": [0, 1, 2, 0, 0],
        }
        df = pds.DataFrame(data)

        # create simple pipeline
        op_df = OpReadDataframe(df)
        pipeline = PipelineDefault("test", [(op_df, {})])

        # create dataset
        dataset = DatasetDefault(data["sample_id"], dynamic_pipeline=pipeline)
        dataset.create()

        # create sampler
        batch_sampler = BatchSamplerDefault(
            dataset, batch_size=3, balanced_class_name="data.class", num_balanced_classes=3, workers=0
        )

        # Use the collate function
        dl = DataLoader(dataset, collate_fn=CollateDefault(), batch_sampler=batch_sampler)
        batch = next(iter(dl))

        # verify
        self.assertEqual(len(batch_sampler), 3)
        self.assertIn(0, batch["data.class"])
        self.assertIn(1, batch["data.class"])
        self.assertIn(2, batch["data.class"])


if __name__ == "__main__":
    unittest.main()
