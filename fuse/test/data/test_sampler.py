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

import unittest
import numpy as np
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from fuse.data.dataset.dataset_wrapper import FuseDatasetWrapper
from fuse.data.sampler.sampler_balanced_batch import FuseSamplerBalancedBatch
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseSamplerBalancedTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # Create dataset
        self.torch_dataset = torchvision.datasets.MNIST('/tmp/mnist_test', download=True, train=True, transform=transform)
        pass

    def test_balanced_dataset(self):
        num_classes = 10
        probs = num_classes / 100
        num_samples = 60000

        # wrapping torch dataset
        dataset = FuseDatasetWrapper(name='test', dataset=self.torch_dataset, mapping=('image', 'label'))
        dataset.create()
        print(dataset.summary(statistic_keys=['data.label']))
        sampler = FuseSamplerBalancedBatch(dataset=dataset,
                                           balanced_class_name='data.label',
                                           num_balanced_classes=num_classes,
                                           batch_size=5,
                                           # balanced_class_weights=[1,1,1,1,1,1,1,1,1,1])  # relevant when batch size is % num classes
                                           balanced_class_probs=[probs for i in range(num_classes)])

        labels = np.zeros(num_classes)

        # Create dataloader
        dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=0)
        iter1 = iter(dataloader)
        for _ in range(len(dataloader)):
            batch_dict = next(iter1)
            labels_in_batch = FuseUtilsHierarchicalDict.get(batch_dict, 'data.label')
            for label in labels_in_batch:
                labels[label] += 1

        # final balance
        print(labels)
        for idx in range(num_classes):
            sampled = labels[idx] / num_samples
            print(f'Class {idx}: {sampled * 100}% of data')
            self.assertAlmostEqual(sampled, probs, delta=probs * 0.5, msg=f'Unbalanced class {idx}, expected 0.1+-0.05 and got {sampled}')

    def test_unbalanced_dataset(self):
        num_classes = 10
        probs = num_classes / 100

        # wrapping torch dataset
        unbalanced_dataset = FuseDatasetWrapper(name='test', dataset=self.torch_dataset, mapping=('image', 'label'))
        unbalanced_dataset.create()

        samples_to_save = []
        stats = [1000, 200, 200, 200, 300, 500, 700, 800, 900, 1000]
        chosen = np.zeros(10)
        for idx in range(60000):
            label = unbalanced_dataset.get(idx, 'data.label')
            if stats[label] > chosen[label]:
                samples_to_save.append(idx)
                chosen[label] += 1

        unbalanced_dataset.samples_description = samples_to_save

        sampler = FuseSamplerBalancedBatch(dataset=unbalanced_dataset,
                                           balanced_class_name='data.label',
                                           num_balanced_classes=num_classes,
                                           batch_size=5,
                                           # balanced_class_weights=[1,1,1,1,1,1,1,1,1,1])  # relevant when batch size is % num classes
                                           balanced_class_probs=[probs for i in range(num_classes)])

        labels = np.zeros(num_classes)

        # Create dataloader
        dataloader = DataLoader(dataset=unbalanced_dataset, batch_sampler=sampler, num_workers=0)
        iter1 = iter(dataloader)
        num_items = 0
        for _ in range(len(dataloader)):
            batch_dict = next(iter1)
            labels_in_batch = FuseUtilsHierarchicalDict.get(batch_dict, 'data.label')
            for label in labels_in_batch:
                labels[label] += 1
                num_items += 1

        # final balance
        print(labels)
        for idx in range(num_classes):
            sampled = labels[idx] / num_items
            print(f'Class {idx}: {sampled * 100}% of data')
            self.assertAlmostEqual(sampled, probs, delta=probs * 0.5, msg=f'Unbalanced class {idx}, expected 0.1(+-0.05) and got {sampled}')

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
