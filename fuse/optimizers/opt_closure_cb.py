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

from typing import Dict
from fuse.managers.callbacks.callback_base import FuseCallback
from fuse.managers.manager_state import FuseManagerState

class FuseCallbackOptClosure(FuseCallback):
    """
    Use this callback if an optimizer requires closure argument
    """
    def on_train_begin(self, state: FuseManagerState):
        self.virtual_batch = []
        self.state = state
        self.state.opt_closure = self.opt_closure

    def on_virtual_batch_begin(self, mode: str, virtual_batch: int):
        self.virtual_batch = []

    def on_batch_end(self, mode: str, batch: int, batch_dict: Dict = None):
        self.virtual_batch.append(batch_dict)

    def opt_closure(self) -> None:
        for batch in self.virtual_batch:
            # forward pass
            batch['model'] = self.state.net(batch)

            # compute loss
            total_loss: torch.Tensor = 0
            for loss_name, loss_function in self.state.losses.items():
                current_loss_result = loss_function(batch)
                # sum all losses for backward
                total_loss += current_loss_result
            total_loss.backward()