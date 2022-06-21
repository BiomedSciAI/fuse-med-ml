import traceback
from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Union
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

                 
class LightningModuleDefault(pl.LightningModule):
    def __init__(self,
                 model_dir: str,
                 model: Optional[torch.nn.Module] = None,
                 losses: Optional[Dict[str, LossBase]] = None,
                 train_metrics: Optional[OrderedDict[str, MetricBase]] = None,
                 validation_metrics: Optional[OrderedDict[str, MetricBase]] = None,
                 test_metrics: Optional[OrderedDict[str, MetricBase]] = None,
                 optimizers_and_lr_schs: Any = None,
                 callbacks: Optional[Sequence[pl.Callback]] = None,
                 best_epoch_source: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs):
        """
        :param optimizers_and_lr_schs: see pl.LightningModule.configure_optimizers for details and relevant options
        """
        super().__init__(**kwargs)

        self._model_dir = model_dir
        self._model = model
        self._losses = losses
        self._metrics = {"train": train_metrics, "validation": validation_metrics, "test": test_metrics, "predict": None}
        self._optimizers_and_lr_schs = optimizers_and_lr_schs
        self._callbacks = callbacks if callbacks is not None else []
        if best_epoch_source is not None:
            self._callbacks += model_checkpoint_callbacks(model_dir, best_epoch_source)
        self._prediction_keys = {}
    
    def set_return_predictions_keys(self, keys: List[str]) -> None:
        if get_sample_id_key() not in keys:
            keys.append(get_sample_id_key())
        
        self._prediction_keys["predict"] = keys

    ## forward
    def forward(self, batch_dict: NDict) -> NDict:
        return self._model(batch_dict)
    
    ## Step
    def training_step(self, batch_dict: NDict, batch_idx: int) -> torch.Tensor:
        return self.step("train", batch_dict, batch_idx)
    
    def validation_step(self, batch_dict: NDict, batch_idx: int) -> torch.Tensor:
        return self.step("validation", batch_dict, batch_idx)
    
    def test_step(self, batch_dict: NDict, batch_idx: int) -> torch.Tensor:
        return self.step("test", batch_dict, batch_idx)

    def predict_step(self, batch_dict: NDict, batch_idx: int) -> torch.Tensor:
        return self.step("predict", batch_dict, batch_idx)
        
    def step(self, mode: str, batch_dict: NDict, batch_idx: int) -> torch.Tensor:
        return_dict = None

        # forward pass
        batch_dict['model'] = self.forward(batch_dict)

        # loss
        # compute total loss and keep loss results
        if self._losses is not None:
            return_dict = {}
            total_loss = None
            for loss_name, loss_function in self._losses.items():
                current_loss_result = loss_function(batch_dict)
                batch_dict['losses.' + loss_name] = current_loss_result.data.item()
                # sum all losses for backward
                if total_loss is None:
                    total_loss = current_loss_result
                else:
                    total_loss += current_loss_result
                
            if total_loss is not None:
                batch_dict['losses.total_loss'] = total_loss.data.item()
                return_dict["loss"] = total_loss
        
            return_dict["losses_values"]=batch_dict["losses"]
            
        # metrics - collect data
        if self._metrics[mode] is not None:
            for _, metric in self._metrics[mode].items():
                # handle batch doesn't return a value, the actual value of the metric is per epoch
                metric.collect(batch_dict)
             
        
        # aggregate values (other than losses)
        if mode in self._prediction_keys:
            outputs = {}
            sample_ids = batch_dict[get_sample_id_key()]
            if isinstance(sample_ids, torch.Tensor):
                sample_ids = list(sample_ids.detach().cpu().numpy())
            outputs['id'] = sample_ids

            for key in self._prediction_keys[mode]:
                output = batch_dict[key]
                if isinstance(output, torch.Tensor):
                    output = output.detach().cpu().numpy()
                outputs[key] = output

            if return_dict is None:
                return_dict = outputs
            else:
                return_dict["predictions"] = outputs
        
        return return_dict
        
    ## Epoch end
    def training_epoch_end(self, step_outputs) -> None:
        return self.epoch_end("train", step_outputs)
    
    def validation_epoch_end(self, step_outputs) -> None:
        return self.epoch_end("validation", step_outputs)

    def epoch_end(self, mode: str, step_outputs: Any) -> None:
        # log losses
        if step_outputs is not None and len(step_outputs) > 0:
            keys = step_outputs[0]["losses_values"].keys()
            for key in keys:
                loss = mean([elem["losses_values"][key] for elem in step_outputs])
                self.log(f"{mode}.losses.{key}", loss, on_epoch=True)
        
        epoch_results = NDict()
        if self._metrics[mode] is not None:
            # compute metrics and keep the results
            for metric_name, metric in self._metrics[mode].items():
                try:
                    metric_result = metric.eval(epoch_results)
                except:
                    track = traceback.format_exc()
                    print(f'Metric {metric_name} process() func failed. Setting results to None')
                    print(track)
                    metric_result = None

                epoch_results[f"metrics.{metric_name}"] = metric_result
                metric.reset()

            # filter per sample results
            for key in epoch_results.keypaths():
                if epoch_results[key] is not None and not isinstance(epoch_results[key], (PerSampleData)):
                    self.log(f"{mode}.{key}", epoch_results[key], on_epoch=True)
            
    # confiugration
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self._optimizers_and_lr_schs

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """
        Override for more custom lr scheduler logic
        """
        super().lr_scheduler_step(scheduler, optimizer_idx, metric)

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        return self._callbacks
    
    