from typing import Dict
import torch
from fuse.utils.ndict import NDict


class LossWrapToDict(torch.nn.Module):
    """
    Wraps a torch loss function to support a batch dict
    """

    def __init__(self, *, loss_module: torch.nn.Module, loss_arg_to_batch_key: Dict[str, str]) -> None:
        """
        :param loss_module: the loss module to wrap
        :param loss_arg_to_batch_key: each key is an argument in the forward function,
                                      each value is the corresponding key in the batch_dict
        """
        super().__init__()
        self._loss_module = loss_module
        self.loss_arg_to_batch_key = loss_arg_to_batch_key

    def forward(self, batch_dict: NDict) -> torch.Tensor:
        # collect arguments for loss module
        loss_kwargs = {arg: batch_dict[batch_key] for arg, batch_key in self.loss_arg_to_batch_key.items()}
        # run loss function
        loss = self._loss_module(**loss_kwargs)
        return loss
