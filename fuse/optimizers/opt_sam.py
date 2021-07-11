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

# slightly modified (error message only) version of https://github.com/davda54/sam/blob/main/sam.py distributed under MIT license
#
# Copyright ©
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from typing import Dict

import torch

from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
from fuse.managers.callbacks.callback_base import FuseCallback
from fuse.managers.manager_state import FuseManagerState


class SAM(torch.optim.Optimizer):
    """
    SAM optimizer - see https://github.com/davda54/sam/blob/main/sam.py
    To use in FuseMedML:
    Create the optimizer and add FuseCallbackSamOpt() to the list of callbacks:.
    Examples:
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay)
    callbacks.append(FuseCallbackSamOpt())

    """
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        # do nothing, instead use first_step and second_step
        return

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


class FuseCallbackSamOpt(FuseCallback):
    """
    Use this callback for SAM optimizer
    """

    def on_train_begin(self, state: FuseManagerState):
        self.state = state

    def on_batch_end(self, mode: str, batch: int, batch_dict: Dict = None):
        # self.virtual_batch.append(batch_dict)
        if mode == 'train':
            self.state.optimizer.first_step(zero_grad=True)
            # forward net
            batch_dict['model'] = self.state.net(batch_dict)
            # compute total loss and keep loss results
            total_loss: torch.Tensor = 0
            for loss_name, loss_function in self.state.losses.items():
                current_loss_result = loss_function(batch_dict)
                FuseUtilsHierarchicalDict.set(batch_dict, 'losses.' + loss_name, current_loss_result.data.item())
                # sum all losses for backward
                total_loss += current_loss_result
            total_loss.backward()
            self.state.optimizer.second_step(zero_grad=True)
