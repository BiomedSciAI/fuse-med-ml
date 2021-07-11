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

import os
from typing import Dict
from fuse.managers.callbacks.callback_base import FuseCallback
from fuse.managers.manager_state import FuseManagerState
from fuse.utils.utils_file import FuseUtilsFile as file
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseTensorboardCallback(FuseCallback):
    """
        Responsible for writing the data of both training and validation to tensorborad loggers under model_dir.
    """

    def __init__(self, model_dir: str) -> None:
        super().__init__()
        self.model_dir = model_dir

    def on_step_end(self, step: int, train_results: Dict = None, validation_results: Dict = None, learning_rate: float = None) -> None:
        """
        Writes the results into train and validation loggers

        :param step: step number
        :param train_results: contains the data to be saved to train tensorboard.
                    e.g.: {'losses': {'loss1': mean_loss1,
                               'loss2': mean_loss2,
                                'total_loss': mean (loss1 + loss2)}
                    'metrics': {'metric1': epoch_metric1,
                                'metric2': epoch_metric2}}
        :param validation_results: contains the data to be saved to train tensorboard.
                    e.g.: {'losses': {'loss1': mean_loss1,
                               'loss2': mean_loss2,
                                'total_loss': mean (loss1 + loss2)}
                    'metrics': {'metric1': epoch_metric1,
                                'metric2': epoch_metric2}}
        :param learning_rate: the learning rate at the end of the step.
        """

        # update train tensorboard logger
        for evaluator_name in FuseUtilsHierarchicalDict.flatten(train_results).keys():
            evaluator_value = FuseUtilsHierarchicalDict.get(train_results, evaluator_name)
            if evaluator_value is not None and not isinstance(evaluator_value, (str,)):
                self.add_scalar(self.tensorboard_logger_train, tag=evaluator_name, scalar_value=evaluator_value, global_step=step)
        self.add_scalar(self.tensorboard_logger_train, 'learning_rate', learning_rate, step)

        # update train tensorboard logger
        if validation_results is not None:
            for evaluator_name in FuseUtilsHierarchicalDict.flatten(validation_results).keys():
                evaluator_value = FuseUtilsHierarchicalDict.get(validation_results, evaluator_name)
                if evaluator_value is not None and not isinstance(evaluator_value, (str,)):
                    self.add_scalar(self.tensorboard_logger_validation, tag=evaluator_name, scalar_value=evaluator_value, global_step=step)
            self.add_scalar(self.tensorboard_logger_validation, 'learning_rate', learning_rate, step)

        return

    def on_train_begin(self, state: FuseManagerState) -> None:
        """
        Called at the beginning of the train procedure.

        :param state: ignored
        """
        # File writer imports are done here in order to workaround the GPU issues -
        # when importing torch.tensorboard cuda gets occupied - do that only AFTER CUDA_VISIBLE_DEVICES is set
        try:
            # available only from torch 1.2
            from torch.utils.tensorboard import SummaryWriter
            self.writer_class = SummaryWriter
            self.use_summary_tf = False
        except ModuleNotFoundError:
            # fallback, use tensorflow file writer
            from tensorflow.summary import FileWriter
            import tensorflow as tf
            self.writer_class = FileWriter
            self.tf_summary = tf.Summary
            self.use_summary_tf = True

        tensorboard_train_dir = os.path.join(self.model_dir, 'train')
        tensorboard_validation_dir = os.path.join(self.model_dir, 'validation')

        # make sure we have these folders
        file.create_dir(tensorboard_train_dir, error_if_exist=False)
        file.create_dir(tensorboard_validation_dir, error_if_exist=False)

        # Get TensorBoard loggers
        self.tensorboard_logger_train = self.writer_class(tensorboard_train_dir)
        self.tensorboard_logger_validation = self.writer_class(tensorboard_validation_dir)
        pass

    def add_scalar(self, writer, tag, scalar_value, global_step):
        """Log a scalar variable."""
        if scalar_value is not None and not isinstance(scalar_value, str):
            if self.use_summary_tf:  # FileWrite
                summary = self.tf_summary(value=[self.tf_summary.Value(tag=tag, simple_value=scalar_value)])
                writer.add_summary(summary, global_step)
            else:
                writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)  # SummaryWriter