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

from abc import abstractmethod
from fuse.utils.ndict import NDict
import torch


class LossBase(torch.nn.Module):
   """
   Base class for Fuse loss functions. 
   Essentially it is torch.nn.Module whose forward method gets as an input batch_dict and returns loss tensor
   Note - if you already have a pytorch implementation of the loss, a useful alternative that you can use is LossDefault class to wrap it

   LossBase Usage example:

   Class MyCustomLoss(LossBase):
         '''
         Extracts prediction and data from batch_dict and performs a dummy loss calculation,
         clamping the loss value to be at most "cap"
         '''
         def __init__(self, 
            pred_name: str = None,
            target_name: str = None,
            cap:float = 0.3,
            ):
            self._pred_name = pred_name
            self._target_name = target_name
            self._cap = cap            

         def forward(self, batch_dict:NDict) -> torch.Tensor:
            #extract pred and target from batch_dict
            pred = batch_dict[self.pred_name]
            target = batch_dict[self.target_name]

            #apply our loss logic, and clamp to cap
            z = torch.mean(pred)-torch.mean(target)
            z = torch.sqrt(z**2) 
            z = torch.clamp(z, max=self._cap) #clamp to cap
            return z

   batch_dict = ... #some batch_dict, containing tensors in keys 'input.preds' and 'groundtruth.targets'
   my_loss = MyCustomLoss(pred_name='input.preds', target_name='groundtruth.target', cap=0.4 )
   loss_val = my_loss(batch_dict)
   ...
   
   """
   @abstractmethod
   def forward(self, batch_dict: NDict) -> torch.Tensor:
      raise NotImplementedError
