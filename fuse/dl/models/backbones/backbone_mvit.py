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

import torch.nn as nn
from torchvision.models.video.mvit import MViT, MViT_V2_S_Weights
from typing import Dict, List
from torch import Tensor
import torch
from torchvision.models.video.mvit import MSBlockConfig


class BackboneMViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,  # can be 1
        input_dim: List[int] = [16, 224, 224],
        pretrained: bool = False,
    ) -> None:
        """
        Multiscale Vision Transformers (MViT) is based on connecting the seminal idea of multiscale feature hierarchies with transformer models.
        This MViT model is based on the MViTv2: Improved Multiscale Vision Transformers for Classification and Detection (https://arxiv.org/abs/2112.01526)
        and Multiscale Vision Transformers papers (https://arxiv.org/abs/2104.11227).
        :param in_channels: number of channels in the image
        :param input_dim: shape of each image in the format [depth, height, width]

        :param pretrained: if True loads the pretrained video mvit_v2 model.
        """
        super(BackboneMViT, self).__init__()
        config: Dict[str, List] = {
            "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
            "input_channels": [
                96,
                96,
                192,
                192,
                384,
                384,
                384,
                384,
                384,
                384,
                384,
                384,
                384,
                384,
                384,
                768,
            ],
            "output_channels": [
                96,
                192,
                192,
                384,
                384,
                384,
                384,
                384,
                384,
                384,
                384,
                384,
                384,
                384,
                768,
                768,
            ],
            "kernel_q": [
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            "kernel_kv": [
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            "stride_q": [
                [1, 1, 1],
                [1, 2, 2],
                [1, 1, 1],
                [1, 2, 2],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 2, 2],
                [1, 1, 1],
            ],
            "stride_kv": [
                [1, 8, 8],
                [1, 4, 4],
                [1, 4, 4],
                [1, 2, 2],
                [1, 2, 2],
                [1, 2, 2],
                [1, 2, 2],
                [1, 2, 2],
                [1, 2, 2],
                [1, 2, 2],
                [1, 2, 2],
                [1, 2, 2],
                [1, 2, 2],
                [1, 2, 2],
                [1, 1, 1],
                [1, 1, 1],
            ],
        }

        block_setting = []
        for i in range(len(config["num_heads"])):
            block_setting.append(
                MSBlockConfig(
                    num_heads=config["num_heads"][i],
                    input_channels=config["input_channels"][i],
                    output_channels=config["output_channels"][i],
                    kernel_q=config["kernel_q"][i],
                    kernel_kv=config["kernel_kv"][i],
                    stride_q=config["stride_q"][i],
                    stride_kv=config["stride_kv"][i],
                )
            )
        self.model = MViT(
            tuple(input_dim[1:]),
            input_dim[0],
            block_setting,
            residual_pool=True,
            residual_with_cls_embed=False,
            rel_pos_embed=True,
            proj_after_attn=True,
            stochastic_depth_prob=0.2,
        )
        if pretrained:
            if input_dim == [16, 224, 224]:
                weights = MViT_V2_S_Weights.DEFAULT
                self.model.load_state_dict(weights.get_state_dict(progress=True))
            else:
                print("pretrained compatible only for input_dim = [16,224,224]")

        if in_channels == 1:
            self.model.conv_proj.in_channels = 1
            self.model.conv_proj.weight = torch.nn.Parameter(
                self.model.conv_proj.weight.sum(dim=1, keepdim=True)
            )
        self.model.head = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
