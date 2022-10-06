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
from torch.utils.data.dataloader import DataLoader

from fuse.dl.lightning.pl_funcs import *  # noqa
from fuse.data.datasets.dataset_base import DatasetBase
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault
from fuse.utils.data.collate import CollateToBatchList


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
        **kwargs,
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
    def training_epoch_end(self, step_outputs) -> None:
        # for the logs to be at each epoch, not each step
        self.log("step", float(self.current_epoch), on_epoch=True, sync_dist=True)
        # calc average epoch loss and log it
        epoch_end_compute_and_log_losses(self, "train", [e["losses"] for e in step_outputs])
        # evaluate  and log it
        epoch_end_compute_and_log_metrics(self, "train", self._train_metrics)

    def validation_epoch_end(self, step_outputs) -> None:
        # for the logs to be at each epoch, not each step
        self.log("step", float(self.current_epoch), on_epoch=True, sync_dist=True)
        # calc average epoch loss and log it
        epoch_end_compute_and_log_losses(self, "validation", [e["losses"] for e in step_outputs])
        # evaluate  and log it
        epoch_end_compute_and_log_metrics(self, "validation", self._validation_metrics)

    def test_epoch_end(self, step_outputs) -> None:
        # for the logs to be at each epoch, not each step
        self.log("step", self.current_epoch, on_epoch=True, sync_dist=True)
        # calc average epoch loss and log it
        epoch_end_compute_and_log_losses(self, "test", [e["losses"] for e in step_outputs])
        # evaluate  and log it
        epoch_end_compute_and_log_metrics(self, "test", self._test_metrics)

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


class BalancedLightningDataModule(pl.LightningDataModule):
    """
    Fuse generic implementation of PL datamodule.
    This module aims to wrap one's datasets into a datamodule for Lightning's Trainer.
    See use case example in "run_mnist_ddp.py"

    For more robust changes consider to implement your datamodule from scratch.
    """

    def __init__(
        self,
        train_dataset: DatasetBase,
        validation_dataset: DatasetBase,
        predict_dataset: DatasetBase,
        num_workers: int,
        batch_size: int,
        balanced_class_name: str,
        num_balanced_classes: int,
        sampler_mode: str = "exact",
        collate_fn: CollateToBatchList = None,
        use_custom_batch_sampler: bool = False,
        verbose: bool = False,
    ):
        """
        :param train_dataset: training dataset
        :param validation_dataset: validation dataset
        :param predict_dataset: prediction (inference) dataset
        :param num_workers: number of processes
        :param batch_size: batch size
        :param balanced_class_name: see BatchSamplerDefault class for ref.  // ASK @MOSHIKO, maybe pass with **batch_sampler_args (?) more nit but less readable (?)
        :param num_balanced_classes: see BatchSamplerDefault class for ref.
        :param sampler_mode: see BatchSamplerDefault class for ref.
        :param collate_fn: dataloader param
        :param use_custom_batch_sampler: set True to use Fuse's BatchSamplerDefault, else will use default one.
                Note that currently DDP doesn't work with our custom batch sampler. Will be fixed in the future.
        :param verbose: set to true for debug messages
        """
        super().__init__()
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._predict_dataset = predict_dataset
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._balanced_class_name = balanced_class_name
        self._num_balanced_classes = num_balanced_classes
        self._sampler_mode = sampler_mode
        self._use_custom_batch_sampler = use_custom_batch_sampler
        self._verbose = verbose
        if collate_fn is None:
            self._collate_fn = CollateDefault()

    def train_dataloader(self):
        """
        returns train dataloader with class custom args
        """
        if self._use_custom_batch_sampler:
            # Create fuse batch sampler and nullify batch_size
            print("Create BatchSamplerDefault:")
            batch_sampler = BatchSamplerDefault(
                dataset=self._train_dataset,
                balanced_class_name=self._balanced_class_name,
                num_balanced_classes=self._num_balanced_classes,
                batch_size=self._batch_size,
                mode=self._sampler_mode,
                verbose=self._verbose,
                workers=self._num_workers,
            )
            print("Create BatchSamplerDefault: DONE")
            batch_size = 1  # should not provide batch_size for custom batch_sampler (1 is default)
        else:
            # Doesn't use custom batch sampler
            batch_sampler = None
            batch_size = self._batch_size  # should provide batch_size for default batch_sampler

        train_dl = DataLoader(
            dataset=self._train_dataset,
            batch_size=batch_size,
            batch_sampler=batch_sampler,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
        )
        return train_dl

    def val_dataloader(self):
        """
        returns validation dataloader with class custom args
        """
        validation_dl = DataLoader(
            dataset=self._validation_dataset,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
        )

        return validation_dl

    def predict_dataloader(self):
        """
        returns prediction dataloader with class custom args
        """
        predict_dl = DataLoader(
            dataset=self._predict_dataset,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
        )
        return predict_dl
