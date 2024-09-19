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
from typing import Optional, Union, Tuple, Callable
from collections import OrderedDict
import os

from fuse.dl.lightning.pl_funcs import *  # noqa
from fuse.utils.file_io.file_io import create_dir


class LightningModuleDefault(pl.LightningModule):
    """
    Generic implementation of LightningModule using FuseMedML style focusing primarily on supervised training.
    FuseMedML conventions make it possible to have such a generic implementation.
    """

    def __init__(
        self,
        model_dir: Optional[str],
        model: torch.nn.Module,
        losses: Optional[Dict[str, LossBase]] = None,
        validation_losses: Optional[List[Tuple[str, Dict[str, LossBase]]]] = None,
        train_metrics: Optional[OrderedDict[str, MetricBase]] = None,
        validation_metrics: Optional[
            Union[
                OrderedDict[str, MetricBase],
                List[Tuple[str, OrderedDict[str, MetricBase]]],
            ]
        ] = None,
        test_metrics: Optional[
            Union[
                OrderedDict[str, MetricBase],
                List[Tuple[str, OrderedDict[str, MetricBase]]],
            ]
        ] = None,
        optimizers_and_lr_schs: Union[dict, Callable] = None,
        callbacks: Optional[Sequence[pl.Callback]] = None,
        best_epoch_source: Optional[Union[Dict, List[Dict]]] = None,
        save_hyperparameters_kwargs: Optional[dict] = None,
        save_model: bool = False,
        save_arguments: bool = False,
        tensorboard_sep: str = ".",
        log_unit: str = None,
        **kwargs: dict,
    ):
        """
        :param model_dir: location for checkpoints and logs
        :param model: Pytorch model to use
        :param losses: dict of FuseMedML style losses
               Will be used for both train and validation unless validation_losses is specified.
        :param validation_losses: Optional, typically used when there are multiple validation dataloaders - each with a different loss
                                List of tuples (must keep the same validation dataloaders order). Each tuple built from validation_dataloader name and the corresponding losses
        :param train_metrics: dict of FuseMedML style metrics - used for training set
        :param validation_metrics: ordereddict of FuseMedML style metrics - used for validation set (must be different instances of metrics (from train_metrics!)
                                   In case of multiple validation dataloaders,  validation_metrics should be list of tuples (that keeps the same dataloaders list order),
                                   Each tuple built from validation dataloader name and corresponding metrics dict.
        :param test_metrics: ordereddict of FuseMedML style metrics - used for test set (must be different instances of metrics (from train_metrics)
                                   In case of multiple test dataloaders, test_metrics should be list of tuples (that keeps the same dataloaders list order),
                                   Each tuple built from test dataloader name and corresponding metrics dict.
        :param optimizers_and_lr_schs: either a callable that follows pl.LightningModule.configure_optimizers prototype or just the output of such a function.
        :param callbacks: see pl.LightningModule.configure_callbacks return value for details
        :param best_epoch_source: Create list of pl.callbacks that saves checkpoints using (pl.callbacks.ModelCheckpoint) and print per epoch summary (fuse.dl.lightning.pl_epoch_summary.ModelEpochSummary).
                                  Either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
        :param save_hyperparameters_kwargs: specify pl.LightningModule.save_hyperparameters() arguments to save the hyper parameters. By default saved all except model_dir and the model (which stored separately)
                To load checkpoint (assuming save_model == True and save_arguments == True) do the following:
                "
                    model_dir = <path/to/the/original/model_dir>
                    checkpoint_path = os.path.join(model_dir, "last_epoch.ckpt")
                    nn_model = torch.load(os.path.join(model_dir, "model.pth"))
                    arguments = torch.load(os.path.join(model_dir, "arguments.pth"))
                    pl_model = LightningModuleDefault.load_from_checkpoint(checkpoint_path=checkpoint_path, model_dir=model_dir, model=nn_model, **arguments)
                "
                To load only the model (assuming save_model == True):
                "
                    model_dir = <path/to/the/original/model_dir>
                    checkpoint_path = os.path.join(model_dir, "last_epoch.ckpt")
                    nn_model = torch.load(os.path.join(model_dir, "model.pth"))
                    pl_model = LightningModuleDefault.load_from_checkpoint(checkpoint_path=checkpoint_path, model_dir=model_dir, model=nn_model)
                "

        :param save_model: save pickled format of the model
        :param save_arguments: save pickled format of main __init__ arguments (not including the model)
        :param tensorboard_sep: use "/" for cleaner tensorboard. "." is for backward compatibility.
        """
        super().__init__(**kwargs)

        if (save_arguments or save_model) and (model_dir is None):
            raise Exception(
                "Error: saving arguments or saving model requires a model_dir to be supplied as well."
            )

        # create model_dir
        if model_dir is not None:
            create_dir(model_dir)

        # save hyper parameters
        if save_hyperparameters_kwargs is not None:
            self.save_hyperparameters(**save_hyperparameters_kwargs)
        else:
            # do nothing - save the arguments once in model dir instead of on every checkpoint
            pass

        # save the model into model_dir - useful in case you want to load it without the original script
        if save_model:
            torch.save(model, os.path.join(model_dir, "model.pth"))

        # save the rest of the arguments - useful in case you want to load it without the original script
        if save_arguments:
            arguments = dict(
                losses=losses,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                optimizers_and_lr_schs=optimizers_and_lr_schs,
                callbacks=callbacks,
                best_epoch_source=best_epoch_source,
            )
            torch.save(arguments, os.path.join(model_dir, "arguments.pth"))

        # store arguments
        self._model_dir = model_dir
        self._model = model
        self._losses = losses if losses is not None else {}
        self._validation_losses = validation_losses

        self._train_metrics = train_metrics if train_metrics is not None else {}
        self._validation_metrics = (
            validation_metrics if validation_metrics is not None else {}
        )
        if test_metrics is None:
            self._test_metrics = self._validation_metrics
        else:
            self._test_metrics = test_metrics
        # convert all use-cases to the same format that supports multiple val dataloaders: List[Tuple[str, OrderedDict[str, MetricBase]]]
        if isinstance(self._validation_metrics, dict):
            self._validation_metrics = [(None, self._validation_metrics)]
        if isinstance(self._test_metrics, dict):
            self._test_metrics = [(None, self._test_metrics)]

        if log_unit not in [None, "optimizer_step", "epoch"]:
            raise Exception(f"Error: unexpected log_unit {log_unit}")

        self._log_unit = log_unit

        self._optimizers_and_lr_schs = optimizers_and_lr_schs
        self._callbacks = callbacks if callbacks is not None else []

        self._callbacks += model_default_callbacks(model_dir, best_epoch_source)

        # init state
        self._prediction_keys = None
        self._sep = tensorboard_sep

        self._training_step_outputs = []

        self._validation_step_outputs = {
            i: [] for i, _ in enumerate(self._validation_metrics)
        }
        self._test_step_outputs = {i: [] for i, _ in enumerate(self._test_metrics)}

    ## forward
    def forward(self, batch_dict: NDict) -> NDict:
        # workaround for fsdp
        if not isinstance(batch_dict, NDict):
            batch_dict = NDict(batch_dict)
        return self._model(batch_dict)

    ## Step
    def training_step(self, batch_dict: NDict, batch_idx: int) -> torch.Tensor:
        # add step number to batch_dict
        batch_dict["global_step"] = self.global_step
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict = self.forward(batch_dict)
        # workaround for fsdp
        if not isinstance(batch_dict, NDict):
            batch_dict = NDict(batch_dict)
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss and save losses values in batch_dict["losses"]
        total_loss = step_losses(self._losses, batch_dict)
        # given the batch_dict and FuseMedML style metrics - collect the required values to compute the metrics on epoch_end
        step_metrics(self._train_metrics, batch_dict)
        # aggregate losses
        self._training_step_outputs.append({"losses": batch_dict["losses"]})
        # return the total_loss
        return total_loss

    def validation_step(
        self, batch_dict: NDict, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        # add step number to batch_dict
        batch_dict["global_step"] = self.global_step
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict = self.forward(batch_dict)
        # workaround for fsdp
        if not isinstance(batch_dict, NDict):
            batch_dict = NDict(batch_dict)
        if self._validation_losses is not None:
            losses = self._validation_losses[dataloader_idx][1]
        else:
            losses = self._losses
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss (ignored) and save losses values in batch_dict["losses"]
        _ = step_losses(losses, batch_dict)
        # given the batch_dict and FuseMedML style metrics - collect the required values to compute the metrics on epoch_end
        step_metrics(self._validation_metrics[dataloader_idx][1], batch_dict)
        # aggregate losses
        if losses:  # if there are losses, collect the results
            self._validation_step_outputs[dataloader_idx].append(
                {"losses": batch_dict["losses"]}
            )

    def test_step(
        self, batch_dict: NDict, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        # add step number to batch_dict
        batch_dict["global_step"] = self.global_step
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict = self.forward(batch_dict)
        # workaround for fsdp
        if not isinstance(batch_dict, NDict):
            batch_dict = NDict(batch_dict)
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss (ignored) and save losses values in batch_dict["losses"]
        if self._validation_losses is not None:
            losses = self._validation_losses[dataloader_idx][1]
        else:
            losses = self._losses

        _ = step_losses(losses, batch_dict)
        # given the batch_dict and FuseMedML style metrics - collect the required values to compute the metrics on epoch_end
        step_metrics(self._test_metrics[dataloader_idx][1], batch_dict)
        # aggregate losses
        if losses:  # if there are losses, collect the results
            self._test_step_outputs[dataloader_idx].append(
                {"losses": batch_dict["losses"]}
            )

    def predict_step(self, batch_dict: NDict, batch_idx: int) -> dict:
        if self._prediction_keys is None:
            raise Exception(
                "Error: predict_step expects list of prediction keys to extract from batch_dict. Please specify it using set_predictions_keys() method "
            )
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict = self.forward(batch_dict)
        # workaround for fsdp
        if not isinstance(batch_dict, NDict):
            batch_dict = NDict(batch_dict)
        # extract the required keys - defined in self.set_predictions_keys()
        return step_extract_predictions(self._prediction_keys, batch_dict)

    ## Epoch end
    def on_train_epoch_end(self) -> None:
        step_outputs = self._training_step_outputs
        # for the logs to be at each epoch, not each step
        if self._log_unit == "epoch":
            self.log("step", float(self.current_epoch), on_epoch=True, sync_dist=True)
        # calc average epoch loss and log it
        epoch_end_compute_and_log_losses(
            self, "train", [e["losses"] for e in step_outputs], sep=self._sep
        )
        # evaluate  and log it
        epoch_end_compute_and_log_metrics(
            self, "train", self._train_metrics, sep=self._sep
        )
        # reset state
        self._training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        step_outputs_lst = self._validation_step_outputs
        # for the logs to be at each epoch, not each step
        if self._log_unit == "epoch":
            self.log("step", float(self.current_epoch), on_epoch=True, sync_dist=True)
        for dataloader_idx, step_outputs in step_outputs_lst.items():
            if len(self._validation_metrics) == 1:
                prefix = "validation"
            else:
                prefix = f"validation.{self._validation_metrics[dataloader_idx][0]}"
            # calc average epoch loss and log it
            epoch_end_compute_and_log_losses(
                self, prefix, [e["losses"] for e in step_outputs], sep=self._sep
            )
            # evaluate  and log it
            epoch_end_compute_and_log_metrics(
                self, prefix, self._validation_metrics[dataloader_idx][1], sep=self._sep
            )
        # reset state
        self._validation_step_outputs = {
            i: [] for i, _ in enumerate(self._validation_metrics)
        }

    def on_test_epoch_end(self) -> None:
        step_outputs_lst = self._test_step_outputs
        # for the logs to be at each epoch, not each step
        if self._log_unit == "epoch":
            self.log("step", float(self.current_epoch), on_epoch=True, sync_dist=True)
        for dataloader_idx, step_outputs in step_outputs_lst.items():
            if len(self._test_metrics) == 1:
                prefix = "test"
            else:
                prefix = f"test.{self._test_metrics[dataloader_idx][0]}"
            # calc average epoch loss and log it
            epoch_end_compute_and_log_losses(
                self, prefix, [e["losses"] for e in step_outputs], sep=self._sep
            )
            # evaluate and log it
            epoch_end_compute_and_log_metrics(
                self, prefix, self._test_metrics[dataloader_idx][1], sep=self._sep
            )
        # reset state
        self._test_step_outputs = {i: [] for i, _ in enumerate(self._test_metrics)}

    # configuration
    def configure_callbacks(self) -> Sequence[pl.Callback]:
        """train loop callbacks"""
        return self._callbacks

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """See pl.LightningModule.configure_optimizers return value for all options"""
        if isinstance(self._optimizers_and_lr_schs, Callable):
            return self._optimizers_and_lr_schs(self)
        return self._optimizers_and_lr_schs

    def set_predictions_keys(self, keys: List[str]) -> None:
        """Define which keys to extract from batch_dict on prediction mode"""
        self._prediction_keys = keys
