import traceback
from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union
from statistics import mean
from fuse.data.utils.sample import get_sample_id_key
from fuse.utils.data.collate import uncollate
import pandas as pd

import torch
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint
from .pl_epoch_summary import ModelEpochSummary

from fuse.utils import NDict
from fuse.dl.losses.loss_base import LossBase
from fuse.eval import MetricBase
from fuse.eval.metrics.utils import PerSampleData



def model_checkpoint_callbacks(model_dir: str, best_epoch_source: Union[Dict, List[Dict]]) -> List[pl.Callback]:
    callbacks = []
    # checkpoints
    if not isinstance(best_epoch_source, list):
        best_epoch_source = [best_epoch_source]
    for checkpoint_to_monitor in best_epoch_source:
        if "dirpath" not in checkpoint_to_monitor:
            checkpoint_to_monitor["dirpath"] = model_dir
        if "filename" not in checkpoint_to_monitor:
            checkpoint_to_monitor["filename"] = "best_epoch"
            if len(best_epoch_source) > 1:  
                checkpoint_to_monitor["auto_insert_metric_name"] = True
          
        model_checkpoint = ModelCheckpoint(
                **checkpoint_to_monitor
        )
        model_checkpoint_display = ModelEpochSummary(dirpath=checkpoint_to_monitor["dirpath"],
                                                          monitor=checkpoint_to_monitor.get("monitor", None),
                                                          mode=checkpoint_to_monitor.get("mode", "min"))
        callbacks.append(model_checkpoint)
        callbacks.append(model_checkpoint_display)
    
    # last epoch checkpoint
    callbacks.append(ModelCheckpoint(dirpath=model_dir, filename="last_epoch", save_last=True))
    return callbacks

def convert_predictions_to_dataframe(predictions: List[NDict]) -> pd.DataFrame:
    """list of batch_dict to a dataframe"""
    assert len(predictions) > 0

    values = {}
    predictions_per_sample = []
    for elem in predictions:
        predictions_per_sample += uncollate(elem)
    keys = predictions_per_sample[0].keypaths()
    for key in keys:
        values[key] = [elem[key] for elem in predictions_per_sample]
    
    df = pd.DataFrame(values)
    return df

def step_losses(losses: Dict[str, LossBase], batch_dict: NDict) -> torch.Tensor:
    total_loss = None
    for loss_name, loss_function in losses.items():
        current_loss_result = loss_function(batch_dict)
        batch_dict['losses.' + loss_name] = current_loss_result.data.item()
        # sum all losses for backward
        if total_loss is None:
            total_loss = current_loss_result
        else:
            total_loss += current_loss_result
        
    if total_loss is not None:
        batch_dict['losses.total_loss'] = total_loss.data.item()
    
    return total_loss

def step_metrics(metrics: OrderedDict[str, MetricBase], batch_dict: NDict) -> None:
    for _, metric in metrics.items():
        # handle batch doesn't return a value, the actual value of the metric is per epoch
        metric.collect(batch_dict)
             

def step_extract_predictions(prediction_keys: Sequence[str], batch_dict: NDict) -> Dict[str, Any]:
    outputs = {}
    sample_ids = batch_dict[get_sample_id_key()]
    if isinstance(sample_ids, torch.Tensor):
        sample_ids = list(sample_ids.detach().cpu().numpy())
    outputs['id'] = sample_ids
    for key in prediction_keys:
        output = batch_dict[key]
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        outputs[key] = output

    return outputs

def epoch_end_compute_and_log_losses(pl: pl.LightningModule, mode: str, batch_losses: Sequence[Dict]) -> None:
    keys = batch_losses[0].keys()
    for key in keys:
        loss = mean([elem[key] for elem in batch_losses])
        pl.log(f"{mode}.losses.{key}", loss, on_epoch=True)

def epoch_end_compute_and_log_metrics(pl: pl.LightningModule, mode: str, metrics: OrderedDict[str, MetricBase]) -> None:
    # compute metrics
    epoch_results = NDict()
    # compute metrics and keep the results
    for metric_name, metric in metrics.items():
        try:
            metric_result = metric.eval(epoch_results)
        except:
            track = traceback.format_exc()
            print(f'Metric {metric_name} process() func failed. Setting results to None')
            print(track)
            metric_result = None

        epoch_results[f"metrics.{metric_name}"] = metric_result
        metric.reset()
    
    # log metrics
    for key in epoch_results.keypaths():
        if epoch_results[key] is not None and not isinstance(epoch_results[key], (PerSampleData)):
            pl.log(f"{mode}.{key}", epoch_results[key], on_epoch=True)

        
                 
class LightningModuleDefault(pl.LightningModule):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

        # should call to self.configure to configure it
        self._model = None
        self._losses = None
        self._train_metrics = None
        self._validation_metrics = None
        self._optimizers_and_lr_schs = None
        self._callbacks = None
        self._prediction_keys = {}
    
   
    ## forward
    def forward(self, batch_dict: NDict) -> NDict:
        return self._model(batch_dict)
    
    ## Step
    def training_step(self, batch_dict: NDict, batch_idx: int) -> torch.Tensor:
        batch_dict["model"] = self.forward(batch_dict)
        total_loss = step_losses(self._losses, batch_dict)
        step_metrics(self._train_metrics, batch_dict)

        return {"loss": total_loss, "losses": batch_dict["losses"]} # return just the losses and drop everything else
     
    def validation_step(self, batch_dict: NDict, batch_idx: int) -> torch.Tensor:
        batch_dict["model"] = self.forward(batch_dict)
        _ = step_losses(self._losses, batch_dict)
        step_metrics(self._validation_metrics, batch_dict)

        return {"losses": batch_dict["losses"]} # return just the losses and drop everything else
     
    def predict_step(self, batch_dict: NDict, batch_idx: int) -> torch.Tensor:
        batch_dict["model"] = self.forward(batch_dict)
        return step_extract_predictions(self._prediction_keys, batch_dict)
        
    ## Epoch end
    def training_epoch_end(self, step_outputs) -> None:
        epoch_end_compute_and_log_losses(self, "train", [e["losses"] for e in step_outputs])
        epoch_end_compute_and_log_metrics(self, "train", self._train_metrics)
    
    def validation_epoch_end(self, step_outputs) -> None:
        epoch_end_compute_and_log_losses(self, "validation", [e["losses"] for e in step_outputs])
        epoch_end_compute_and_log_metrics(self, "validation", self._validation_metrics)
    
    # confiugration
    def configure(self):
        self._model = self.configure_models()
        self._losses = self.configure_losses()
        self._train_metrics = self.configure_train_metrics()
        self._validation_metrics = self.configure_train_metrics()
        # self.configure_callbacks() will be called by trainer
        # self.configure_optimizers() will be called by trainer

    def configure_optimizers(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        raise NotImplementedError
    
    def configure_models(self) -> None:
        raise NotImplementedError
    
    def configure_train_metrics(self) -> None:
        raise NotImplementedError
    
    def configure_validation_metrics(self) -> None:
        raise NotImplementedError
    
    def configure_losses(self) -> None:
        raise NotImplementedError
    
    def set_predictions_keys(self, keys: List[str]) -> None:
        self._prediction_keys = keys
    
    def set_predictions_keys(self, keys: List[str]) -> None:
        self._prediction_keys = keys