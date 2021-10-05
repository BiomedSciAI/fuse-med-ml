from typing import Dict, Union
from fuse.metrics.metric_base import FuseMetricBase
from fuse.losses.loss_base import FuseLossBase

class FuseMetricFromLoss(FuseMetricBase):
    def __init__(self, loss: FuseLossBase):
        super().__init__(pred_name=None, target_name=None) # no need to collect
        self._loss = loss
        self._epoch_losses = []
    
    def collect(self, batch_dict: Dict) -> None:
        loss = self._loss.forward(batch_dict).data.item()
        self._epoch_losses.append(loss)
    
    def reset(self) -> None:
        self._epoch_losses = []
    
    def process(self) -> Union[float, Dict[str, float], str]:
        if len(self._epoch_losses) == 0:
            return 0.0
        return sum(self._epoch_losses) / len(self._epoch_losses)