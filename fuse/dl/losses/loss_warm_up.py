import torch


class LossWarmUp(torch.nn.Module):
    """
    Zero the loss until a defined amount of iterations have passed.

    This is useful for example when you have multiple losses and you want one loss to stabilize before the other is used
    """

    def __init__(self, loss: torch.nn.Module, nof_iterations: int):
        super().__init__()
        self._loss = loss
        self._nof_iterrations = nof_iterations
        self._count = 0

    def forward(self, *args, **kwargs):
        if self._count < self._nof_iterrations:
            self._count += 1
            return torch.tensor(0.0)
        else:
            return self._loss.forward(*args, **kwargs)
