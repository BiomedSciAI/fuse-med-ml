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

import pytorch_lightning as pl
from typing import Optional

from fuse.dl.lightning.pl_funcs import *  # noqa


class LightningModuleDefault(pl.LightningModule):
    """
    Generic implementation of LightningModule using FuseMedML style focusing primarily on supervised training.
    FuseMedML conventions make it possible to have such a generic implementation.
    """

    def __init__(
        self,
        model_dir: str,
        model: Optional[torch.nn.Module] = None,
        losses: Optional[Dict[str, LossBase]] = None,
        train_metrics: Optional[OrderedDict[str, MetricBase]] = None,
        validation_metrics: Optional[OrderedDict[str, MetricBase]] = None,
        test_metrics: Optional[OrderedDict[str, MetricBase]] = None,
        optimizers_and_lr_schs: Any = None,
        callbacks: Optional[Sequence[pl.Callback]] = None,
        best_epoch_source: Optional[Union[Dict, List[Dict]]] = None,
        save_hyperparameters: Optional[List[str]] = None,
        tensorboard_sep: str = ".",
        **kwargs: dict,
    ):
        """
        :param model_dir: location for checkpoints and logs
        :param model: Pytorch model to use
        :param optimizers_and_lr_schs: see pl.LightningModule.configure_optimizers for details and relevant options
        :param losses: dict of FuseMedML style losses
        :param train_metrics: dict of FuseMedML style metrics - used for training set
        :param validation_metrics: dict of FuseMedML style metrics - used for validation set (must be different instances of metrics (from train_metrics!)
        :param test_metrics: dict of FuseMedML style metrics - used for test set (must be different instances of metrics (from train_metrics and validation_metrics!)
        :param optimizers_and_lr_schs: see pl.LightningModule.configure_optimizers return value for all options
        :param callbacks: see pl.LightningModule.configure_callbacks return value for details
        :param best_epoch_source: Create list of pl.callbacks that saves checkpoints using (pl.callbacks.ModelCheckpoint) and print per epoch summary (fuse.dl.lightning.pl_epoch_summary.ModelEpochSummary).
                                  Either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
        :param save_hyperparameters: specify which hyperparameters you would like to save. Default None.  See pl.LightningModule.save_hyperparameters() for more details.
        :param tensorboard_sep: use "/" for cleaner tensorboard. "." is for backward compatibility.
        """
        super().__init__(**kwargs)
        if save_hyperparameters is not None:
            self.save_hyperparameters(*save_hyperparameters)

        # store arguments
        self._model_dir = model_dir
        self._model = model
        self._losses = losses if losses is not None else {}
        self._train_metrics = train_metrics if train_metrics is not None else {}
        self._validation_metrics = validation_metrics if validation_metrics is not None else {}
        self._test_metrics = test_metrics if test_metrics is not None else {}

        self._optimizers_and_lr_schs = optimizers_and_lr_schs
        self._callbacks = callbacks if callbacks is not None else []
        if best_epoch_source is not None:
            self._callbacks += model_checkpoint_callbacks(model_dir, best_epoch_source)

        # init state
        self._prediction_keys = None
        self._sep = tensorboard_sep

    ## forward
    def forward(self, batch_dict: NDict) -> NDict:
        return self._model(batch_dict)

    ## Step
    def training_step(self, batch_dict: NDict, batch_idx: int) -> dict:
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict = self.forward(batch_dict)
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss and save losses values in batch_dict["losses"]
        total_loss = step_losses(self._losses, batch_dict)
        # given the batch_dict and FuseMedML style losses - collect the required values to compute the metrics on epoch_end
        step_metrics(self._train_metrics, batch_dict)
        # return the total_loss, the losses and drop everything else
        return {"loss": total_loss, "losses": batch_dict["losses"]}

    def validation_step(self, batch_dict: NDict, batch_idx: int) -> dict:
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict = self.forward(batch_dict)
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss (ignored) and save losses values in batch_dict["losses"]
        _ = step_losses(self._losses, batch_dict)
        # given the batch_dict and FuseMedML style losses - collect the required values to compute the metrics on epoch_end
        step_metrics(self._validation_metrics, batch_dict)

        # return just the losses and drop everything else
        return {"losses": batch_dict["losses"]}

    def test_step(self, batch_dict: NDict, batch_idx: int) -> dict:
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict = self.forward(batch_dict)
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss (ignored) and save losses values in batch_dict["losses"]
        _ = step_losses(self._losses, batch_dict)
        # given the batch_dict and FuseMedML style losses - collect the required values to compute the metrics on epoch_end
        step_metrics(self._test_metrics, batch_dict)

        # return just the losses and drop everything else
        return {"losses": batch_dict["losses"]}

    def predict_step(self, batch_dict: NDict, batch_idx: int) -> dict:
        if self._prediction_keys is None:
            raise Exception(
                "Error: predict_step expects list of prediction keys to extract from batch_dict. Please specify it using set_predictions_keys() method "
            )
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict = self.forward(batch_dict)
        # extract the required keys - defined in self.set_predictions_keys()
        return step_extract_predictions(self._prediction_keys, batch_dict)

    ## Epoch end
    def training_epoch_end(self, step_outputs: List[dict]) -> None:
        # for the logs to be at each epoch, not each step
        self.log("step", float(self.current_epoch), on_epoch=True, sync_dist=True)
        # calc average epoch loss and log it
        epoch_end_compute_and_log_losses(self, "train", [e["losses"] for e in step_outputs], sep=self._sep)
        # evaluate  and log it
        epoch_end_compute_and_log_metrics(self, "train", self._train_metrics, sep=self._sep)

    def validation_epoch_end(self, step_outputs: List[dict]) -> None:
        # for the logs to be at each epoch, not each step
        self.log("step", float(self.current_epoch), on_epoch=True, sync_dist=True)
        # calc average epoch loss and log it
        epoch_end_compute_and_log_losses(self, "validation", [e["losses"] for e in step_outputs], sep=self._sep)
        # evaluate  and log it
        epoch_end_compute_and_log_metrics(self, "validation", self._validation_metrics, sep=self._sep)

    def test_epoch_end(self, step_outputs: List[dict]) -> None:
        # for the logs to be at each epoch, not each step
        self.log("step", self.current_epoch, on_epoch=True, sync_dist=True)
        # calc average epoch loss and log it
        epoch_end_compute_and_log_losses(self, "test", [e["losses"] for e in step_outputs], sep=self._sep)
        # evaluate  and log it
        epoch_end_compute_and_log_metrics(self, "test", self._test_metrics, sep=self._sep)

    # configuration
    def configure_callbacks(self) -> Sequence[pl.Callback]:
        """train loop callbacks"""
        return self._callbacks

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """See pl.LightningModule.configure_optimizers return value for all options"""
        return self._optimizers_and_lr_schs

    def set_predictions_keys(self, keys: List[str]) -> None:
        """Define which keys to extract from batch_dict on prediction mode"""
        self._prediction_keys = keys
