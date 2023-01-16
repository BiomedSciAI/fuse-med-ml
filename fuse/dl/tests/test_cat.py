"""
(C) Copyright 2023 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on Jan 11, 2023

"""

import torch
import unittest
from fuse.dl.models.backbones.backbone_transformer import CrossAttentionTransformer


class TestCrossAttentionTransformer(unittest.TestCase):
    def test_all_contexts(self) -> None:

        # model parameters
        model_params = {
            "emb_dim": 128,
            "num_tokens_a": 10000,
            "num_tokens_b": 20000,
            "max_seq_len_a": 512,
            "max_seq_len_b": 1024,
            "output_dim": 256,
        }

        for context in ["seq_a", "seq_b", "both"]:
            model_params["context"] = context
            self.validate_model_with_params(model_params)

    def validate_model_with_params(self, model_params: dict) -> None:
        """
        Basic testing for the CrossAttentionTransformer model
        """

        # init model
        model = CrossAttentionTransformer(**model_params)

        # init random sequences that don't exceed max sequences length
        s1 = torch.randint(0, model_params["num_tokens_a"], (1, 400))
        s2 = torch.randint(0, model_params["num_tokens_b"], (1, 1000))

        # processing sample
        output = model(s1, s2)

        # validation
        assert output.shape[0] == 1
        if output.shape[1] != model_params["output_dim"]:
            raise Exception(
                f"Expected output dimension to be {model_params['output_dim']}, but got: {output.shape[1]}. model parameters are: {model_params}."
            )


if __name__ == "__main__":
    unittest.main()
