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

import torch

from fuse.dl.losses.loss_wrap_to_dict import usage_example


class TestViT(unittest.TestCase):
    def test_vit_usage_example(self) -> None:
        loss = usage_example()
        self.assertEqual(type(loss), torch.Tensor)


if __name__ == "__main__":
    unittest.main()
