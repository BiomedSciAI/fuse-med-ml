from typing import Dict
import torch
from torch import nn
from fuse.utils.ndict import NDict


class LossWrapToDict(torch.nn.Module):
    """
    Wraps a torch loss function to support a batch dict.
    See usage_example() below for additional details.
    """

    def __init__(
        self, *, loss_module: torch.nn.Module, loss_arg_to_batch_key: Dict[str, str], weight: float = 1.0
    ) -> None:
        """
        :param loss_module: the loss module to wrap
        :param loss_arg_to_batch_key: each key is an argument in the forward function,
                                      each value is the corresponding key in the batch_dict
        """
        super().__init__()
        self._loss_module = loss_module
        self.loss_arg_to_batch_key = loss_arg_to_batch_key
        self._weight = weight

    def forward(self, batch_dict: NDict) -> torch.Tensor:
        # collect arguments for loss module
        loss_kwargs = {arg: batch_dict[batch_key] for arg, batch_key in self.loss_arg_to_batch_key.items()}
        # run loss function
        loss = self._loss_module(**loss_kwargs)
        return self._weight * loss


def usage_example() -> torch.Tensor:
    batch_size = 100
    num_classes = 10
    # creating an example of a batch_dict
    logits = torch.randn(size=(batch_size, num_classes))
    gt = torch.randint(high=num_classes, size=(batch_size,))
    batch_dict = NDict({"data.gt": gt, "model.logits": logits})

    # creating the loss
    loss = nn.CrossEntropyLoss()
    # the CrossEntropyLoss forward function has two arguments: input, target
    # this maps every forward argument to the relevant key in the batch_dict
    loss_arg_to_batch_key = {"input": "model.logits", "target": "data.gt"}
    # wrapping the loss
    wrapped_loss = LossWrapToDict(loss_module=loss, loss_arg_to_batch_key=loss_arg_to_batch_key)
    # running the wrapped loss with the batch_dict
    loss_result = wrapped_loss(batch_dict)
    print(f"the output loss is: {loss_result}")
    return loss_result


if __name__ == "__main__":
    usage_example()
