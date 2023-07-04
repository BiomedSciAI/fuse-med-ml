import torch

from fuse.utils import NDict


class LossExtractFromBatchDict(torch.nn.Module):
    """
    Read the loss from batch dict
    Useful in cases that the model is the one that set the loss
    """

    def __init__(self, key: str, weight: float = 1.0) -> None:
        """
        :param key: batch_dict key to read the loss from
        :param weight: loss weight, the returned value will be multiplication of loss and weight
        """
        super().__init__()
        self.key = key
        self.weight = weight

    def forward(self, batch_dict: NDict) -> torch.Tensor:
        return batch_dict[self.key] * self.weight
