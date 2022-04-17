from typing import Dict
import torch

from fuse.metrics.metric_base import FuseMetricBase

class FuseLossWarmUp(torch.nn.Module):
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
        