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

from typing import Dict, List


from typing import Dict, Any, List, Iterator, Optional, Union, Sequence, Hashable, Callable
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from fuse.losses.loss_base import FuseLossBase
from fuse.metrics.metric_base import FuseMetricBase


class FuseManagerState:
    """
    FuseManagerState contains the current state of the manager.
    """

    def __init__(self) -> None:

        self.output_model_dir: str

        self.net: nn.Module
        self.metrics: Dict[str, FuseMetricBase] = {}
        self.losses: Dict[str, FuseLossBase] = {}
        self.optimizer: Optimizer
        self.lr_scheduler: Any
        self.train_params: Dict = {}
        self.opt_closure: Optional[Callable] = None
        # self.ensemble_nets: Sequence[nn.Module]
        self.num_gpus: int
        self.device: str

        # number of epochs:
        self.num_epochs: int
        self.end_epoch: int
        self.current_epoch: int

        # number of batches in a virtual batch - user defined
        self.virtual_batch_size: int

        # updated throughout run
        self.learning_rate: float

        # best epoch
        # user definitions
        self.best_epoch_source: List[Dict[str, str]]
        self.best_epoch_function: List[str] = []
        self.optimization_function: List[str] = []
        self.on_equal_values: List[str] = []
        self.num_models_to_save: int
        # updated throughout run
        self.best_epoch: List[int]
        self.best_epoch_values: List[Dict] = []

        # checkpoint saving user definitions
        self.gap_between_saving_epochs: int
        self.start_saving_epochs: int