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

from fuse.managers.manager_state import FuseManagerState


class FuseCallback(object):
    """
        Abstract base class used to build new callbacks.
        Callbacks are called at various stages during training and infer.

    """

    def __init__(self):
        pass

    def on_step_begin(self, step: int) -> None:
        """
        In general, a step is the part of the train that: handle_epoch for both train and validation data, update the scheduler and save checkpoint.
        The step number is the epoch number.

        Called on step begin.
        :param step: step number
        """
        pass

    def on_step_end(self, step: int, train_results: Dict = None, validation_results: Dict = None, learning_rate: float = None) -> None:
        """
        In general, a step is the part of the train that:
        handle_epoch for both train and validation data, update the scheduler and save checkpoint.
        The step number is the epoch number.

        Called on step end.
        :param step: step number
        :param train_results: contains train losses and metrics.
                    e.g.: {
                        'losses': {'cls_loss': 0.53242,
                                   'total_loss': 0.53242},
                        'metrics' : {'auc' : 0.8676}
                        }
        :param validation_results: contains validation losses and metrics.
                    e.g.: {
                        'losses': {'cls_loss': 0.53242,
                                   'total_loss': 0.53242},
                        'metrics' : {'auc' : 0.8676}
                        }
        :param learning_rate: the learning rate at the end of the step.
        """
        pass

    def on_epoch_begin(self, mode: str, epoch: int) -> None:
        """
        Called at the beginning of epoch.

        :param mode: either 'train', 'validation' or 'infer'
        :param epoch: epoch number
        """
        pass

    def on_epoch_end(self, mode: str, epoch: int, epoch_results: Dict = None) -> None:
        """
        Called at the end of epoch handling.

        :param mode: either 'train', 'validation' or 'infer'
        :param epoch: epoch number
        :param epoch_results: hierarchical dictionary where keys are losses and metrics computed on data.
            E.g.: {
                    'losses': {'loss1': mean_loss1,
                               'loss2': mean_loss2,
                                'total_loss': mean (loss1 + loss2)}
                    'metrics': {'metric1': epoch_metric1,
                                'metric2': epoch_metric2}
               }
            in 'infer' mode it is an empty dict {}
        """
        pass

    def on_virtual_batch_begin(self, mode: str, virtual_batch: int) -> None:
        """
        Called at the beginning of virtual_batch.

        :param mode: either 'train', 'validation' or 'infer'
        :param virtual_batch: virtual batch number
        """
        pass

    def on_virtual_batch_end(self, mode: str, virtual_batch: int, virtual_batch_results: Dict = None) -> None:
        """

        :param mode: either 'train', 'validation' or 'infer'
        :param virtual_batch: virtual batch number
        :param virtual_batch_results: hierarchical dictionary with the computed losses
                    'losses': {'loss1': mean_loss1,
                               'loss2': mean_loss2,
                               'total_loss': mean (loss1 + loss2)}
        """
        pass

    def on_batch_begin(self, mode: str, batch: int) -> None:
        pass

    def on_data_fetch_end(self, mode: str, batch: int, batch_dict: Dict = None) -> None:
        """
        Called after iter.next() is called.

        :param mode: either 'train', 'validation' or 'infer'
        :param batch: batch number
        :param batch_dict: dict with the retrieved data:
            it's key is 'data' and it contains gt, input and descriptor keys
        """
        pass

    def on_batch_end(self, mode: str, batch: int, batch_dict: Dict = None) -> None:
        """
        Called at the end of handling batch.

        :param mode: either 'train', 'validation' or 'infer'
        :param batch: batch number
        :param batch_dict:
            hierarchical dictionary contains the keys:
            data - with gt, input and descriptor keys.
            losses - key for each computed loss + total_loss.
        """
        pass

    def on_train_begin(self, state: FuseManagerState) -> None:
        """
        Called at the beginning of the train procedure, after initialization of all variables and model.

        :param state: manager state object.
            Contains the state of the manager. For details, see FuseManagerState.
        """
        pass

    def on_train_end(self) -> None:
        pass
