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

from typing import Union

import torch
import torch.nn as nn


class FuseCheckpoint():
    def __init__(self, net: Union[nn.Module, dict], epoch_idx: int, learning_rate: float):
        if isinstance(net, nn.Module):
            self.net_state_dict = net.state_dict()
        else:
            # we have a state dict
            self.net_state_dict = net
        self.epoch_idx = epoch_idx
        self.learning_rate = learning_rate
        pass

    def as_dict(self):
        return {"net_state_dict": self.net_state_dict,
                "epoch_idx": self.epoch_idx,
                "learning_rate": self.learning_rate}

    def save_to_file(self, file_name: str):
        torch.save(self.as_dict(), file_name)

    @classmethod
    def load_from_file(cls, file_name: str):
        checkpoint_dict = torch.load(file_name, map_location='cpu')
        net_state_dict = checkpoint_dict['net_state_dict']
        epoch_idx = checkpoint_dict['epoch_idx']
        learning_rate = checkpoint_dict['learning_rate']

        return cls(net=net_state_dict, epoch_idx=epoch_idx, learning_rate=learning_rate)
