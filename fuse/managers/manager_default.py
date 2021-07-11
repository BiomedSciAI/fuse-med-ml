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

import logging
import os
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import trange, tqdm
from typing import Dict, Any, List, Iterator, Optional, Union, Sequence, Hashable, Callable

from fuse.data.data_source.data_source_base import FuseDataSourceBase
from fuse.data.dataset.dataset_base import FuseDatasetBase
from fuse.data.processor.processor_base import FuseProcessorBase
from fuse.data.visualizer.visualizer_base import FuseVisualizerBase
from fuse.losses.loss_base import FuseLossBase
from fuse.managers.callbacks.callback_base import FuseCallback
from fuse.managers.callbacks.callback_debug import FuseCallbackDebug
from fuse.managers.callbacks.callback_infer_results import FuseInferResultsCallback
from fuse.managers.manager_state import FuseManagerState
from fuse.metrics.metric_base import FuseMetricBase
from fuse.models.model_ensemble import FuseModelEnsemble
from fuse.utils import utils_misc as misc
from fuse.utils.utils_checkpoint import FuseCheckpoint
from fuse.utils.utils_debug import FuseUtilsDebug
from fuse.utils.utils_file import FuseUtilsFile as file
from fuse.utils.utils_gpu import FuseUtilsGPU as gpu
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
from fuse.utils.utils_logger import log_object_input_state
from fuse.utils.utils_misc import FuseUtilsMisc


class FuseManagerDefault:
    """
    Default implementation of manager. The manager is the main API to use when running Fuse.
    Supports Train and Infer functionality.


    Possible Work flows for using the manager are (see function documentations for parameters description):
    For train:
        FuseManagerDefault() -> manager.set_objects() -> manager.train()
    For Resume training:
        FuseManagerDefault() -> manager.load_objects() -> manager.load_checkpoint() -> manager.train()
    For Train using existing model:
        FuseManagerDefault() -> manager.set_objects() [-> manager.load_objects()] [-> manager.load_checkpoint()] -> manager.train()
    For Infer:
        FuseManagerDefault() -> manager.infer()
        or -
        FuseManagerDefault() -> manager.load_objects() -> manager.load_checkpoint() -> manager.infer()
    For Infer given model:
        FuseManagerDefault() -> manager.set_objects() -> manager.load_checkpoint() -> manager.infer()
    """

    def __init__(self, output_model_dir: str = None, force_reset: bool = False):
        """
        Initialization of the manager.
        Responsible for creating the class members, creating the output_model_dir if not None.

        :param output_model_dir: path to the model directory
        :param force_reset: When False (default) and model_dir content has to be reset the user is prompt to do so.
            When True, the directory is automatically reset.
        """
        log_object_input_state(self, locals())
        self.logger = logging.getLogger('Fuse')

        self.state = FuseManagerState()
        self.state.output_model_dir = output_model_dir
        self.state.current_epoch = 0

        if output_model_dir is not None:
            # prepare model_dir
            file.create_or_reset_dir(output_model_dir, ignore_files=['logs', 'source_files'], force_reset=force_reset)

        self.callbacks: List[FuseCallback] = list()  # callback can be empty
        pass

    def set_objects(self,
                    net: nn.Module = None,
                    # ensemble_nets: Sequence[nn.Module] = None,
                    metrics: Dict[str, FuseMetricBase] = None,
                    losses: Dict[str, FuseLossBase] = None,
                    callbacks: List[FuseCallback] = None,
                    optimizer: Optimizer = None,
                    lr_scheduler: Any = None,
                    best_epoch_source: Union[List[Dict[str, str]], Dict[str, str]] = None,
                    train_params: Dict = None,
                    output_model_dir: str = None) -> None:
        """
        Sets objects to the given values.
        :param net: the model definition
        :param metrics: a dictionary of metric name and FuseMetricBase to compute on train and validation sets.
        :param losses: definition of loss functions
        :param callbacks: list of callbacks to call during train or infer.
        :param optimizer: optimizer to use
        :param lr_scheduler: learning rate scheduler to use
        :param best_epoch_source: metric or loss function to use when deciding on the best epoch. may also be a list of them
                        contain the keys:
                         'source': name of loss or metric function - e.g. losses.cls_loss or metrics.auc
                         'optimization': the optimization function used on the source function. either min or max.
                         'on_equal_values': in case where the current value is equal the best so far value should it be considered as best epoch?
                                            value can be either better or worse.
        :param train_params: dictionary containing configuration data (mostly for training).
            may contain the keys (otherwise filled with defaults):
            device - options: 'cuda', 'cpu', 'cuda:0', ... (default 'cuda')
            num_epochs - number of epochs to run (default 100),
            virtual_batch_size - number of batches in one virtual batch (default 1),
            start_saving_epochs - first epoch to start saving checkpoint (default 80)
            gap_between_saving_epochs - number of epochs between each saved checkpoint
        :param output_model_dir: directory to save the model data to

        """
        best_epoch_source = [best_epoch_source] if not isinstance(best_epoch_source, list) else best_epoch_source

        if net is not None: self.state.net = net
        # if ensemble_nets is not None: self.state.ensemble_nets = ensemble_nets
        if metrics is not None: self.state.metrics = metrics
        if losses is not None: self.state.losses = losses
        if callbacks is not None: self.callbacks = callbacks
        if optimizer is not None: self.state.optimizer = optimizer
        if lr_scheduler is not None: self.state.lr_scheduler = lr_scheduler
        if best_epoch_source is not None: self.state.best_epoch_source = best_epoch_source
        if train_params is not None: self.state.train_params = train_params
        if output_model_dir is not None: self.state.output_model_dir = output_model_dir

        # debug mode - append debug callback
        if FuseUtilsDebug().get_setting('manager_stages') != 'default':
            self.callbacks.append(FuseCallbackDebug())
            self.logger.info(f'Manager - debug mode - append debug callback', {'color': 'red'})
        pass

    def _save_objects(self, validation_dataloader: DataLoader) -> None:
        """
        Saves objects using torch.save (net, losses, metrics, best_epoch_source, optimizer, lr_scheduler, callbacks).
        Each parameter is saved into a separate file (called losses.pth, metrics.pth, etc) under self.output_model_dir.
        :param validation_dataloader: dataloader to extract dataset definitions from (saved on inference_dataset.pth)
        """

        def _torch_save(parameter_to_save: Any, parameter_name: str) -> None:
            file_path = os.path.join(self.state.output_model_dir, f'{parameter_name}.pth')
            torch.save(parameter_to_save, file_path)
            self.logger.debug(f"Saved object to file {file_path}", {'color': 'green'})
            pass

        _torch_save(self.state.net.module, 'net')
        _torch_save(self.state.metrics, 'metrics')
        _torch_save(self.state.losses, 'losses')
        _torch_save(self.callbacks, 'callbacks')
        _torch_save(self.state.optimizer, 'optimizer')
        _torch_save(self.state.lr_scheduler, 'lr_scheduler')
        _torch_save(self.state.best_epoch_source, 'best_epoch_source')
        _torch_save(self.state.train_params, 'train_params')

        # also save validation_dataset in inference mode
        if validation_dataloader is not None:
            FuseDatasetBase.save(validation_dataloader.dataset, mode=FuseDatasetBase.SaveMode.INFERENCE,
                                 filename=os.path.join(self.state.output_model_dir, "inference_dataset.pth"))
        pass

    def load_objects(self, input_model_dir: Union[str, Sequence[str]], list_of_object_names: List[str] = None, mode: str = 'infer') -> Dict[str, Any]:
        """
        Loads objects from torch saved pth files under input_model_dir.
        Note: if mode is infer, only loads net.
        :param input_model_dir: the path to the files to load. if list/tuple: load multiple modules for ensemble inference
        :param list_of_object_names: list of the objects to load. If None, then all objects are loaded.
            For the possible values of objects to be loaded, see input parameters of setObjects().
        :param mode: either 'infer' or 'train'. When 'infer' (default), only net is loaded.
        :return: for each loaded object there's a key and the value is the member variable
        """

        def should_load(object_name):
            return list_of_object_names is None or object_name in list_of_object_names

        def load_if_exists(object_name, force=False):
            file_path = os.path.join(input_model_dir, f'{object_name}.pth')
            if os.path.exists(file_path):
                object_to_set = torch.load(file_path, map_location='cpu')
                loaded_objects.update({object_name: object_to_set})
                return object_to_set
            if force:
                self.logger.error(f"Cannot load this file: {file_path}")
                raise Exception(f"Cannot load this file: {file_path}")

        loaded_objects = {}

        if isinstance(input_model_dir, (tuple, list)):  # if multiple model dirs, load ensemble modules
            if mode != 'infer':
                msg = "Error in load_objects: using multiple model dirs is only supported in 'infer' mode"
                self.logger.error(msg)
                raise Exception(msg)

            # log an ensemble of model dirs
            self.logger.info('Loading ensemble of %d models.' % len(input_model_dir))
            for model_idx, model_dir in enumerate(input_model_dir):
                self.logger.info("Loading ensemble model %d from: %s" % (model_idx, model_dir))

            self.state.net = FuseModelEnsemble(input_model_dir)
            input_model_dir = input_model_dir[0]

        else:  # load single module
            self.logger.info("Loading model from: %s" % input_model_dir)
            self.state.net = load_if_exists('net', force=True) if should_load('net') else self.state.net

        # if mode is infer, then we need only load the net
        if mode == 'infer':
            load_if_exists('inference_dataset') if should_load('inference_dataset') else self.state.metrics
            return loaded_objects

        self.state.metrics = load_if_exists('metrics') if should_load('metrics') else self.state.metrics
        self.state.losses = load_if_exists('losses') if should_load('losses') else self.state.losses
        self.callbacks = load_if_exists('callbacks') if should_load('callbacks') else self.callbacks
        self.state.optimizer = load_if_exists('optimizer') if should_load('optimizer') else self.state.optimizer
        self.state.lr_scheduler = load_if_exists('lr_scheduler') if should_load('lr_scheduler') else self.state.lr_scheduler
        self.state.best_epoch_source = load_if_exists('best_epoch_source') if should_load('best_epoch_source') else self.state.best_epoch_source
        self.state.train_params = load_if_exists('train_params') if should_load('train_params') else self.state.train_params

        # train
        return loaded_objects

    def load_checkpoint(self,
                        checkpoint: Union[str, int, Sequence[Union[str, int]]],
                        model_dir: Optional[Union[str, Sequence[str]]] = None,
                        values_to_resume: List[str] = None,
                        mode: str = 'infer', strict: bool = True,
                        index: int = 0) -> None:
        """
        Loads values saved on checkpoint file.
        When mode = 'infer' loads only the model.

        :param checkpoint: checkpoint definition(s) to load weights from.
                           Possible values are 'best', 'last', epoch number (int) or full path to checkpoint file.
                           Can be a list, in case of an ensemble
        :param model_dir: source folder(s). might be ignored if checkpoint is a full path. Can be a list.
        :param values_to_resume:
            can contain any of: ['net', 'start_epoch', 'learning_rate'].
            when None (default) all values saved to checkpoint are loaded.
        :param mode: either 'infer' (default) or 'train'
        :param strict: input param to net.load_state_dict
        :param index: multiple checkpoint may be saved as best epoch, index indicates which one to take
        """

        def should_load(object_name):
            return values_to_resume is None or object_name in values_to_resume

        # possible values: epoch number, 'last', 'best', or full path to checkpoint file
        checkpoints = [checkpoint] if not isinstance(checkpoint, (list, tuple)) else checkpoint

        if model_dir is not None:
            model_dirs = [model_dir] if not isinstance(model_dir, (list, tuple)) else model_dir

            if len(model_dirs) > 1 and len(checkpoints) == 1:
                checkpoints = checkpoints * len(model_dirs)

        checkpoint_objs = []

        for idx in range(len(checkpoints)):
            checkpoint_desc = checkpoints[idx]
            # if checkpoint_desc is [int, 'best', 'last'], convert it to full path
            if isinstance(checkpoint_desc, int) or (isinstance(checkpoint_desc, str) and (checkpoint_desc in ['best', 'last'])):
                if isinstance(checkpoint_desc, str) and (checkpoint_desc == 'best'):
                    checkpoint_desc = checkpoint_desc + '_' + str(index)
                checkpoint_file = os.path.join(model_dirs[idx], f"checkpoint_{checkpoint_desc}_epoch.pth")
            # otherwise, checkpoint_desc is already a full path to checkpoint
            elif isinstance(checkpoint, str):
                checkpoint_file = checkpoint_desc
            else:
                msg = "Wrong checkpoint definition in 'load_checkpoint'. Possible values: integer/'best'/'last'/full path, but got: %s" % str(
                    checkpoint_desc)
                self.logger.error(msg)
                raise Exception(msg)
            str_vals = 'all' if values_to_resume is None else str(values_to_resume)
            self.logger.info(f'Loading checkpoint file: {checkpoint_file}. values_to_resume {str_vals}', {'color': 'yellow'})
            checkpoint_objs.append(FuseCheckpoint.load_from_file(checkpoint_file))

        if should_load('net'):
            net_state_dict_list = [checkpoint.net_state_dict for checkpoint in checkpoint_objs]
            self.state.net.load_state_dict(*net_state_dict_list, strict=strict)

        if mode == 'train':
            checkpoint_obj = checkpoint_objs[0]
            # on infer mode we don't need to load starting epoch or learning rate
            if should_load('start_epoch'):
                self.state.current_epoch = checkpoint_obj.epoch_idx
                self.logger.info(f'Loaded start epoch: {self.state.current_epoch}', {'color': 'yellow'})

            if should_load('learning_rate'):
                self.state.learning_rate = checkpoint_obj.learning_rate
                for param_group in self.state.optimizer.param_groups:
                    param_group['lr'] = self.state.learning_rate
                self.logger.info(f'Loaded learning rate: {self.state.learning_rate}', {'color': 'yellow'})

        pass

    def train(self, train_dataloader: DataLoader, validation_dataloader: DataLoader = None) -> None:
        """
        Train the net using train dataloader and validation on validation dataloader.

        :param train_dataloader: training data to use
        :param validation_dataloader: validation data to use.
            When None (default), validation is skipped and the best epoch values are decided by the train metrics.
        """
        # check that all objects needed are there

        self._verify_all_objects_initialized(mode='train')

        # debug - num workers
        override_num_workers = FuseUtilsDebug().get_setting('manager_override_num_dataloader_workers')
        if override_num_workers != 'default':
            train_dataloader.num_workers = override_num_workers
            validation_dataloader.num_workers = override_num_workers
            self.logger.info(f'Manager - debug mode - override dataloader num_workers to {override_num_workers}', {'color': 'red'})

        # prepare to use on GPUs
        self.state.net.to(self.state.device)
        self.state.net = nn.DataParallel(self.state.net)

        # TODO move losses to device as well
        total_param = sum(p.numel() for p in self.state.net.parameters())
        trainable_param = sum(p.numel() for p in self.state.net.parameters() if p.requires_grad)
        self.logger.info(f"Total number of parameters in model:{total_param:,}, trainable parameters:{trainable_param:,}",
                         {'color': 'red', 'attrs': 'bold'})

        # save model and parameters for future use (e.g., infer or resume_from_weights)
        self._save_objects(validation_dataloader)

        # save datasets summary into file and logger
        self._handle_dataset_summaries(train_dataloader, validation_dataloader)

        # handle callbacks
        for callback in self.callbacks: callback.on_train_begin(self.state)

        # validation handle_epoch, to see initial state of net
        if validation_dataloader is not None:
            initial_results = self.handle_epoch('validation', 0, validation_dataloader)
        else:
            initial_results = self.handle_epoch('validation', 0, train_dataloader)
        self.state.best_epoch_values = [initial_results for i in range(self.state.num_models_to_save)]
        self.state.current_epoch += 1

        # loop over num of epochs
        while self.state.current_epoch < self.state.end_epoch:
            for callback in self.callbacks: callback.on_step_begin(self.state.current_epoch)

            # train epoch
            self.logger.info(f"Start training on epoch {self.state.current_epoch}")
            train_results = self.handle_epoch('train', self.state.current_epoch, train_dataloader)

            # validation epoch - only if validation is needed
            if validation_dataloader is not None:
                self.logger.info(f"Start validation on epoch {self.state.current_epoch}")
                validation_results = self.handle_epoch('validation', self.state.current_epoch, validation_dataloader)
            else:
                validation_results = None

            epoch_checkpoint = FuseCheckpoint(self.state.net.module.state_dict(), self.state.current_epoch, self.get_current_learning_rate())

            # if this is the best epoch yet
            for i in range(self.state.num_models_to_save):
                if self._is_best_epoch_so_far(train_results, validation_results, i):
                    best_val = FuseUtilsHierarchicalDict.get(self.state.best_epoch_values[i], self.state.best_epoch_function[i])
                    self.logger.info(f"This is the best epoch ever ({self.state.best_epoch_function[i]} = {best_val})",
                                     {'color': 'green', 'attrs': 'bold'})
                    self.state.best_epoch[i] = self.state.current_epoch
                    best_epoch_checkpoint_filename = os.path.join(self.state.output_model_dir, 'checkpoint_best_' + str(i) + '_epoch.pth')
                    epoch_checkpoint.save_to_file(best_epoch_checkpoint_filename)
                # output to screen
                self._write_epoch_summary_table(train_results, validation_results, i)
            # save checkpoint to last epoch file
            last_epoch_checkpoint_filename = os.path.join(self.state.output_model_dir, 'checkpoint_last_epoch.pth')
            epoch_checkpoint.save_to_file(last_epoch_checkpoint_filename)

            if self.is_epoch_for_save(self.state.current_epoch):
                this_epoch_checkpoint_filename = os.path.join(self.state.output_model_dir, f'checkpoint_{self.state.current_epoch}_epoch.pth')
                epoch_checkpoint.save_to_file(this_epoch_checkpoint_filename)

            # LR scheduler update and log
            self.update_scheduler(train_results, validation_results)

            for callback in self.callbacks:
                callback.on_step_end(self.state.current_epoch, train_results, validation_results, self.get_current_learning_rate())

            self.state.current_epoch += 1

        for callback in self.callbacks: callback.on_train_end()

        pass

    def visualize(self, visualizer: FuseVisualizerBase, data_loader: Optional[DataLoader] = None, infer_processor: Optional[FuseProcessorBase] = None,
                  descriptors: Optional[List[Hashable]] = None, device: str = 'cuda', display_func: Optional[Callable] = None):

        """
        Visualize data including the input and the output.
        Expected Sequence:
        1. Using a loaded model to extract the output:
         manager = FuseManagerDefault()

         manager.load_objects(<model dir>, mode='infer')  # this method can load either a single model or an ensemble
         manager.load_checkpoint(checkpoint=<path to checkpoint file>, mode='infer')
         manager.visualize(visualizer=visualizer,
                  data_loader=dataloader,
                  descriptors=<optional - list of descriptors>,
                  display_func=<optional - display_func>,
                  infer_processor=None)

        2. using inference processor
         manager = FuseManagerDefault()
         manager.visualize(visualizer=visualizer,
                  data_loader=dataloader,
                  descriptors=<optional - list of descriptors>,
                  display_func=<optional - display_func>,
                  infer_processor=infer_processor)

        :param visualizer: The visualizer, getting a batch_dict as an input and doing it's magic
        :param data_loader: data loader as used for validation / training / inference
        :param infer_processor: Optional, if specified this function will not run the model and instead extract the output from infer processor
        :param descriptors: Optional. List of sample descriptors, if None will go over the entire dataset. Might be also list of dataset indices.
        :param device: options: 'cuda', 'cpu', 'cuda:0', ... (default 'cuda')
        :param display_func: Function getting the batch dict as an input and returns boolean specifying if to visualize this sample or not.
        :return: None
        """
        dataset: FuseDatasetBase = data_loader.dataset
        if infer_processor is None:
            if not hasattr(self, 'net') or self.state.net is None:
                self.logger.error(f"Cannot visualize without either net or infer_processor")
                raise Exception(f"Cannot visualize without either net or infer_processor")

            # prepare net
            self.state.net.to(device)
            self.state.net = nn.DataParallel(self.state.net)

        if descriptors is None:
            descriptors = range(len(dataset))
        for desc in tqdm(descriptors):
            # extract sample
            batch_dict = dataset.get(desc)
            if infer_processor is None:
                # apply model in case infer processor is not specified
                # convert dimensions to batch
                batch_dict = dataset.collate_fn([batch_dict])
                # run model
                batch_dict['model'] = self.state.net(batch_dict)
                # convert dimensions back to single sample
                FuseUtilsHierarchicalDict.apply_on_all(batch_dict, FuseUtilsMisc.squeeze_obj)
            else:
                # get the sample descriptor of the sample
                sample_descriptor = FuseUtilsHierarchicalDict.get(batch_dict, 'data.descriptor')
                # get the infer data
                infer_data = infer_processor(sample_descriptor)
                # add infer data to batch_dict
                for key in infer_data:
                    FuseUtilsHierarchicalDict.set(batch_dict, key, infer_data[key])

            if display_func is None or display_func(batch_dict):
                visualizer.visualize(batch_dict)

    def infer(self, input_model_dir: Optional[Union[str, Sequence]] = None,
              checkpoint: Optional[Union[str, int, Sequence[Union[str, int]]]] = None,
              data_source: Optional[FuseDataSourceBase] = None, data_loader: Optional[DataLoader] = None,
              num_workers: Optional[int] = 4, batch_size: Optional[int] = 2,
              output_columns: List[str] = None, output_file_name: str = None, strict: bool = True,
              append_default_inference_callback: bool = True,
              checkpoint_index: int = 0) -> pd.DataFrame:
        """
        Inference of net on data. Either the data_source or data_loader should be defined.
        When data_source is defined, validation_dataset is loaded from the original model_dir and is used to create a dataloader.
        Returns the inference Results as dict:
                {
                'descriptor': [id_1, id_2, ...],
                'output': {'output_head1': [res1_1, res1_2, ...],
                           'output_head2': [res2_1, res2_2, ..]}
                }

        ================================================================================================================
         Ensemble Mode
        ===============
        Optionally, run inference using an ensemble of several models. This is done by providing a list of
        model dirs in the 'input_model_dir' param and optionally a list of checkpoints in the 'checkpoint' param.
        When using an ensemble, the first model dir is used for loading Fuse data pipeline: Dataset and Input Processors.
        All model dirs, including the first, are used for loading modules and their checkpoints for the ensemble.
        In this mode, inference Results dict will look like:
                {
                'descriptor': [id_1, id_2, ...],
                'ensemble_output_0': {'output_head1': [res1_1, res1_2, ...],     # first ensemble model
                                      'output_head2': [res2_1, res2_2, ..]},
                'ensemble_output_1': {'output_head1': [res1_1, res1_2, ...],     # second ensemble model
                                      'output_head2': [res2_1, res2_2, ..]},
                ...

                }
        ================================================================================================================

        :param input_model_dir: model directory(s) to load data from
        :param checkpoint: checkpoint definition(s) to load weights from. Possible values are 'best', 'last' or int.
            When not None, model weights are loaded from respected checkpoint file under input_model_dir
            (either checkpoint_best_epoch.pth, checkpoint_last_epoch.pth or checkpoint_{checkpoint}_epoch.pth)
            when None, no checkpoint is loaded (assumes that the weights were already loaded.
            in ensemble mode, can provide either one checkpoint for all models or a sequence of separate checkpoints for each.
        :param data_source: data source to use
        :param data_loader: data loader to use
        :param num_workers: number of processes for Dataloader, effective only if 'data_loader' param is None
        :param batch_size: batch size for Dataloader, effective only if 'data_loader' param is None
        :param output_columns: output columns to return.
            When None (default) all columns are returned.
            When not None, FuseInferResultsCallback callback is created.
        :param output_file_name: output file path. when None (default) results are not saved to file.
        :param strict: strict state dict loading when loading checkpoint weights. default is True.
        :param append_default_inference_callback: if True, appends Fuse's default results collector callback
        :param checkpoint_index: few best checkpoints can be saved, each with its own index
        :return: infer results in a DataFrame
        """

        # debug - num workers
        override_num_workers = FuseUtilsDebug().get_setting('manager_override_num_dataloader_workers')
        if override_num_workers != 'default':
            num_workers = override_num_workers
            if data_loader is not None:
                data_loader.num_workers = override_num_workers
            self.logger.info(f'Manager - debug mode - override dataloader num_workers to {override_num_workers}', {'color': 'red'})

        if input_model_dir is not None:
            # user provided model dir(s), and Manager has no 'net' attribute - need to load modules
            if not hasattr(self.state, 'net'):
                self.load_objects(input_model_dir, mode='infer', list_of_object_names=['net'])  # this method can load either a single model or an ensemble

        if checkpoint is not None:
            if hasattr(self.state, 'net'):
                if isinstance(self.state.net, torch.nn.DataParallel):
                    raise Exception("Error in infer - Manager has a DataParallel net. Cannot load checkpoint into DataParallel module!")
                if input_model_dir is None:
                    msg = "Cannot load checkpoint file without a definition of input_model_dir"
                    self.logger.error(msg)
                    raise Exception(msg)

                self.load_checkpoint(checkpoint=checkpoint, model_dir=input_model_dir, mode='infer', strict=strict, index=checkpoint_index)

        # checklist check that all objects needed are there
        self._verify_all_objects_initialized(mode='infer')

        #TODO I don't like this flag - maybe think about a way to get rid of it?
        # append inference callback
        if append_default_inference_callback:
            self.callbacks.append(FuseInferResultsCallback(output_file=output_file_name, output_columns=output_columns))

        # either optional_datasource or optional_dataloader
        if data_loader is not None and data_source is not None:
            self.logger.error('Cannot have both data_loader and data_source defined')
            raise Exception('Cannot have both data_loader and data_source defined')
        if data_loader is None and data_source is None:
            self.logger.error('Either data_loader or data_source should be defined')
            raise Exception('Either data_loader or data_source should be defined')

        if data_loader is None:
            # need to create a data loader
            # first check that we have the model dir to get these data from
            if input_model_dir is None:
                self.logger.error('Missing parameter input_model_dir! Cannot load data_set from previous model.')
                raise Exception('Missing parameter input_model_dir! Cannot load data_set from previous model.')

            if isinstance(input_model_dir, (tuple, list)):
                data_set_filename = os.path.join(input_model_dir[0], "inference_dataset.pth")
            else:
                data_set_filename = os.path.join(input_model_dir, "inference_dataset.pth")
            self.logger.info(f"Loading data source definitions from {data_set_filename}", {'color': 'yellow'})
            infer_dataset = FuseDatasetBase.load(filename=data_set_filename, override_datasource=data_source)
            data_loader = DataLoader(dataset=infer_dataset, shuffle=False, drop_last=False, batch_sampler=None,
                                     batch_size=batch_size, num_workers=num_workers, collate_fn=infer_dataset.collate_fn)

        # prepare net
        self.state.net.to(self.state.device)
        self.state.net = nn.DataParallel(self.state.net)

        # we are ready to run inference
        self.handle_epoch('infer', 0, data_loader)

        # if infer CB is in the callback list, then return its result
        for callback in self.callbacks:
            if isinstance(callback, FuseInferResultsCallback):
                return callback.get_infer_results()

    def handle_epoch(self, mode: str, epoch: int, data_loader: DataLoader) -> Dict:
        """
        For each virtual batch calls handle_virtual_batch() and aggregates the results into a dict.
        In case mode is 'infer', the prediction results are returned.
        in case mode is 'train' or 'validation', the losses are returned.
        :param mode: mode to run the epoch. Can be either 'train', 'validation', 'infer'
        :param epoch: epoch number
        :param data_loader: data to load
        :return: a hierarchical dictionary depending on the epoch mode.
            For infer:
                epoch_results = {
                            'descriptor': [id_1, id_2, ...],
                            'output': {'output_head1': [res1_1, res1_2, ...],
                                        'output_head2': [res2_1, res2_2, ..]}
                            }
            For Train or Validation:
               epoch_results = {
                            'losses': {'loss1': mean_loss1,
                            '          'loss2': mean_loss2},
                            'metrics': {'metric1': epoch_metric1,
                                        'metric2': epoch_metric2}
               }
        """

        for callback in self.callbacks: callback.on_epoch_begin(mode=mode, epoch=epoch)
        assert mode in ['train', 'validation', 'infer']

        if mode == 'train':
            with torch.enable_grad():
                self.state.net.train()
                epoch_results = self.do_handle_epoch(mode, epoch, data_loader)
                for callback in self.callbacks:
                    callback.on_epoch_end(mode, epoch, epoch_results)
                return epoch_results
        elif mode in ['validation', 'infer']:
            with torch.no_grad():
                self.state.net.eval()
                epoch_results = self.do_handle_epoch(mode, epoch, data_loader)
                for callback in self.callbacks:
                    callback.on_epoch_end(mode, epoch, epoch_results)
                return epoch_results

    def do_handle_epoch(self, mode: str, epoch: int, data_loader: DataLoader) -> Dict:
        """
        Utility function that is called by handle_epoch
        :param mode: mode to run the epoch. Can be either 'train', 'validation', 'infer'
        :param epoch: epoch number
        :param data_loader: data to load
        :return: a dictionary depending on the epoch mode.
            For Train or Validation:
               epoch_results = {

               epoch_results = {
                            'losses': {'loss1': mean_loss1,
                                      'loss2': mean_loss2,
                                      'total_loss': mean_loss1 + mean_loss2},
                            'metrics': {'metric1': epoch_metric1,
                                        'metric2': epoch_metric2}
                                                       }
            For infer returns an empty dict {}.
        """
        # handle each virtual batch separately
        num_batches = int(np.ceil(len(data_loader) / self.state.virtual_batch_size))
        data_iter = iter(data_loader)

        # loop over batches (can be virtual batch)
        epoch_results = {}
        for virtual_batch in trange(num_batches):
            # handle_virtual batch
            virtual_batch_dict = self.handle_virtual_batch(mode, virtual_batch, data_iter)
            epoch_results = _extend_results_dict(mode, virtual_batch_dict, epoch_results)

        # compute metrics and keep the results
        for metric_name, metric in self.state.metrics.items():
            try:
                metric_result = metric.process()
            except:
                lgr = logging.getLogger('Fuse')
                track = traceback.format_exc()
                lgr.error(f'Metric {metric_name} process() func failed. Setting results to None')
                lgr.error(track)
                metric_result = None

            FuseUtilsHierarchicalDict.set(epoch_results, "metrics." + metric_name, metric_result)
            metric.reset()

        # average losses into mean_loss
        if 'losses' in epoch_results:
            for loss in FuseUtilsHierarchicalDict.get_all_keys(epoch_results['losses']):
                batch_losses = FuseUtilsHierarchicalDict.get(epoch_results, 'losses.' + loss)
                loss_mean = np.nansum(batch_losses) / len(batch_losses)
                FuseUtilsHierarchicalDict.set(epoch_results, "losses." + loss, loss_mean)

        return epoch_results

    def handle_virtual_batch(self, mode: str, virtual_batch: int, data_iter: Iterator) -> Dict:
        """
        Responsible for splitting into mini batches, running each mini batch, aggregating its results, and running optimizer at the end

        :param mode: mode to run the epoch. Can be either 'train', 'validation', 'infer'
        :param virtual_batch: virtual batch number
        :param data_iter: iterator of data to use
        :return: a hierarchical dictionary with data of virtual batch, depending on the mode.
            For infer:
                epoch_results = {
                            'descriptor': [id_1, id_2, ...],
                            'output': {'output_head1': [res1_1, res1_2, ...],
                                        'output_head2': [res2_1, res2_2, ..]}
                            }
            For Train or Validation:
               epoch_results = {
                            'losses' : {'loss1': [0.21, 0.45362],
                                        'loss2': [0.01, 0.443],
                                        'total_loss': [0.22, 0.89662]
                                        }
               }
        """
        for callback in self.callbacks: callback.on_virtual_batch_begin(mode=mode, virtual_batch=virtual_batch)

        # mode is train/validation/infer
        if mode == 'train':
            self.state.optimizer.zero_grad()

        virtual_batch_results = {}

        for mini_batch in range(self.state.virtual_batch_size):
            mini_batch_result_dict = self.handle_batch(mode, mini_batch, data_iter)
            virtual_batch_results = _extend_results_dict(mode, mini_batch_result_dict, virtual_batch_results)

        if mode == 'train':
            # after all virtual mini batches all processed, we can run the optimizer
            self.state.optimizer.step(closure=self.state.opt_closure)

        for callback in self.callbacks: callback.on_virtual_batch_end(mode, virtual_batch, virtual_batch_results)

        return virtual_batch_results

    def handle_batch(self, mode: str, batch: int, data_iter: Iterator) -> Dict:
        """Handles the batch load, net forward and metrics computations
        :param mode: mode to run the epoch. Can be either 'train', 'validation', 'infer'
        :param batch: batch number
        :param data_iter: data to load
        :return: hierarchical batch_dict containing:
            descriptor: unique identifier for each sample processed,
            model.output: a dict with keys for each possible output,
            losses: a dict with a key for each defined loss + a total_loss key
        """
        # callbacks handling
        for callback in self.callbacks: callback.on_batch_begin(mode, batch)

        # get the input
        try:
            batch_dict = next(data_iter)
        # in case this was called from the last virtual batch, and we don't have any more inputs
        except StopIteration:
            # callbacks handling
            for callback in self.callbacks: callback.on_batch_end(mode, batch, {})
            return {}

        for callback in self.callbacks: callback.on_data_fetch_end(mode, batch, batch_dict)

        # move every tensor in input to device
        FuseUtilsHierarchicalDict.apply_on_all(batch_dict, gpu.move_tensor_to_device, self.state.device)

        # forward net
        batch_dict['model'] = self.state.net(batch_dict)

        # compute total loss and keep loss results
        total_loss: torch.Tensor = 0
        for loss_name, loss_function in self.state.losses.items():
            current_loss_result = loss_function(batch_dict)
            FuseUtilsHierarchicalDict.set(batch_dict, 'losses.' + loss_name, current_loss_result.data.item())
            # sum all losses for backward
            total_loss += current_loss_result
        # no need to add total_loss if there are no losses computed
        if isinstance(total_loss, torch.Tensor):
            FuseUtilsHierarchicalDict.set(batch_dict, 'losses.total_loss', total_loss.data.item())

        if mode == 'train':
            # backward
            total_loss.backward()

        for callback in self.callbacks: callback.on_batch_end(mode, batch, batch_dict=batch_dict)

        # compute metrics
        for metric_name, metric in self.state.metrics.items():
            # handle batch doesn't return a value, the actual value of the metric is per epoch
            metric.collect(batch_dict)


        return batch_dict

    def update_scheduler(self, train_results: Dict, validation_results: Dict) -> None:
        """
        Update the scheduler using the input parameters. this function is called at the end of each epoch run.
        If the learning rate has changed during the step function, the log is updated accordingly.

        :param train_results: hierarchical dict train epoch results.
            contains the keys: losses, metrics.
                losses is a dict where values are the commputed mean loss for each loss.
                    and an additional key 'total_loss' which is the mean total loss of the epoch.
                metrics is a dict where values are the computed metrics.
        :param validation_results: hierarchical validation epoch results dict.
             contains the keys: losses, metrics.
                losses is a dict where values are the commputed mean loss for each loss.
                    and an additional key 'total_loss' which is the mean total loss of the epoch.
                metrics is a dict where values are the computed metrics.
                Note, if validation was not done on the epoch, this parameter can be None
        """
        # implementation for the train ReduceLROnPlateau scheduler:
        prev_lr = self.get_current_learning_rate()

        # take total_loss from train_results
        self.state.lr_scheduler.step(np.mean(FuseUtilsHierarchicalDict.get(train_results, 'losses.total_loss')))

        curr_lr = self.get_current_learning_rate()

        # log scheduler changes - if lr has changed during step
        if prev_lr != curr_lr:
            self.logger.info(f"Learning rate has changed from {prev_lr} to {curr_lr}")

    def _verify_all_objects_initialized(self, mode: str):
        def verify_value(self_object, parameter):
            if self_object is None:
                self.logger.error(f"Cannot {mode} without {parameter} definition")
                raise Exception(f"Cannot {mode} without {parameter} definition")

        verify_value(self.state.net, 'net')
        verify_value(self.callbacks, 'callbacks')

        if mode == 'train':
            verify_value(self.state.metrics, 'metrics')
            verify_value(self.state.losses, 'losses')
            verify_value(self.state.optimizer, 'optimizer')
            verify_value(self.state.lr_scheduler, 'lr_scheduler')
            verify_value(self.state.train_params, 'train_params')
            verify_value(self.state.best_epoch_source, 'best_epoch_source')
            verify_value(self.state.output_model_dir, 'output_model_dir')

        # make sure we have all the parameters we need
        full_config = self.set_config_defaults(self.state.train_params, mode)

        self.state.virtual_batch_size: int = full_config['virtual_batch_size']

        if mode == 'train':
            self.state.num_epochs: int = full_config['num_epochs']
            # debug - num epochs
            override_num_epochs = FuseUtilsDebug().get_setting('manager_override_num_epochs')
            if override_num_epochs != 'default':
                self.state.num_epochs = override_num_epochs
                self.logger.info(f'Manager - debug mode - override num_epochs to {self.state.num_epochs}', {'color': 'red'})

            self.state.start_saving_epochs: int = full_config['start_saving_epochs']
            self.state.gap_between_saving_epochs: int = full_config['gap_between_saving_epochs']
            self.state.end_epoch: int = self.state.num_epochs

            self.state.num_models_to_save = 1 if isinstance(self.state.best_epoch_source, dict) else len(self.state.best_epoch_source)
            self.state.best_epoch = [0 for _ in range(self.state.num_models_to_save)]
            for i in range(self.state.num_models_to_save):
                # update best epoch source members
                if self.state.metrics:
                    self.state.best_epoch_function.append(
                        self.state.best_epoch_source[i].get('source', f'metrics.{list(self.state.metrics.keys())[0]}'))
                else:
                    self.state.best_epoch_function.append(
                        self.state.best_epoch_source[i].get('source', f'losses.{list(self.state.losses.keys())[0]}'))

                self.state.optimization_function.append(self.state.best_epoch_source[i].get('optimization', 'max'))
                assert self.state.optimization_function[i] in ['max', 'min']
                initial_value = -float('inf') if self.state.optimization_function[i] == 'max' else float('inf')
                self.state.best_epoch_values.append({self.state.best_epoch_function[i]: initial_value})

                on_equal_values = self.state.best_epoch_source[i].get('on_equal_values', 'better')
                assert on_equal_values in ['better', 'worse']
                self.state.on_equal_values.append(on_equal_values)

        self.state.device: str = full_config.get('device')
        pass

    def is_epoch_for_save(self, epoch: int) -> bool:
        """
        Checks whether this epoch should be saved according to the save epoch parameters
        :param epoch: epoch num
        :return: True if the epoch checkpoint should be saved, False Otherwise.
        """

        # save every gap_between_saving_epochs epoch starting from the start_saving_epochs epoch
        return epoch >= self.state.start_saving_epochs and (epoch - self.state.start_saving_epochs) % self.state.gap_between_saving_epochs == 0

    def _is_best_epoch_so_far(self, train_results: Dict, validation_results: Dict, epoch_source_index: int) -> bool:
        """
        Returns true whether the current results are the best results by now.
        if validation results are None, use train_results to decide.
        Uses the definitions of self.state.best_epoch_source in order to decide.
        :param train_results: train results
        :param validation_results:  validation results
        :return: True if results are best so far.
        """

        def is_better_epoch_value(current_value: float) -> bool:
            try:
                current_best = FuseUtilsHierarchicalDict.get(self.state.best_epoch_values[epoch_source_index],
                                                             self.state.best_epoch_function[epoch_source_index])
                # check if we can compare values
                if current_value is None:
                    lgr = logging.getLogger('Fuse')
                    lgr.error(f'Comparing epochs failed since value is None, assuming it is not the best epoch')
                    lgr.error(traceback.format_exc())
                    return False

                # first handle equal values
                if current_value == current_best:
                    return self.state.on_equal_values[epoch_source_index] == 'better'

                # for different values: if max is better
                if self.state.optimization_function[epoch_source_index] == 'max':
                    return current_value > current_best
                # if min is better
                return current_value < current_best
            except:
                lgr = logging.getLogger('Fuse')
                track = traceback.format_exc()
                lgr.error(f'Comparing epochs failed, assuming it is not the best epoch')
                lgr.error(track)
                return False

        if validation_results is not None:
            values_to_check = validation_results
        else:
            values_to_check = train_results
        function_key = self.state.best_epoch_function[epoch_source_index]
        if not FuseUtilsHierarchicalDict.is_in(values_to_check, function_key):
            lgr = logging.getLogger('Fuse')
            lgr.error(f"source function {function_key} does not exist in results_dict. " + \
                      f"Possible values are {FuseUtilsHierarchicalDict.get_all_keys(values_to_check)}")
            lgr.error(traceback.format_exc())
            raise KeyError(f"source function {function_key} does not exist in results_dict." + \
                           f"Possible values are {FuseUtilsHierarchicalDict.get_all_keys(values_to_check)}")

        value_to_compare = FuseUtilsHierarchicalDict.get(values_to_check, function_key)
        if is_better_epoch_value(value_to_compare):
            self.state.best_epoch_values[epoch_source_index] = values_to_check
            return True
        return False

    def get_current_learning_rate(self) -> float:
        """
        Returns the current learning rate used by the optimizer
        :return: learning rate
        """
        return self.state.optimizer.param_groups[0]['lr']

    def _write_epoch_summary_table(self, train_dict: dict, validation_dict: dict, epoch_source_index: int) -> None:
        def get_value_as_float_str(dict, key):
            val_as_str = 'N/A'
            try:
                value = FuseUtilsHierarchicalDict.get(dict, key)
                val_as_str = '%.4f' % float(value)
            except:
                pass
            return val_as_str

        stats_table = pd.DataFrame(columns=['', 'Best Epoch Value', 'Current Epoch Validation', 'Current Epoch Train'])
        idx = 0

        eval_keys = sorted(FuseUtilsHierarchicalDict.get_all_keys(train_dict))
        for evaluator_name in eval_keys:
            train_value_str = get_value_as_float_str(train_dict, evaluator_name)
            validation_val_str = get_value_as_float_str(validation_dict, evaluator_name)
            best_so_far_str = get_value_as_float_str(self.state.best_epoch_values[epoch_source_index], evaluator_name)

            stats_table.loc[idx] = [f'{evaluator_name}', best_so_far_str, validation_val_str, train_value_str]
            idx += 1

        best_source = self.state.best_epoch_function[epoch_source_index]
        if self.state.current_epoch == self.state.best_epoch[epoch_source_index]:
            epoch_title = f'Stats for epoch: {self.state.current_epoch} (Currently the best epoch for source {best_source}!)'
            self.logger.info(epoch_title, {'attrs': ['underline', 'bold']})
        else:
            best_so_far = self.state.best_epoch[epoch_source_index]
            epoch_title = f'Stats for epoch: {self.state.current_epoch} (Best epoch is {best_so_far} for source {best_source})'
            self.logger.info(epoch_title)
        stats_as_str = misc.get_pretty_dataframe(stats_table)
        self.logger.info(stats_as_str)
        self.logger.info(f"Model: {self.state.output_model_dir}")

        try:
            op = 'w' if epoch_source_index == 0 else 'a'  # we want to have all stats in one file per epoch
            with open(os.path.join(self.state.output_model_dir, "last_epoch_summary.txt"), op) as sfile:
                sfile.write(epoch_title)
                sfile.write(stats_as_str)
        except Exception as error:
            self.logger.error(f"Cannot write epoch summary to file {self.state.output_model_dir}/last_epoch_summary.txt")
            self.logger.error(error)
        pass

    def set_config_defaults(self, config: dict, mode: str) -> Dict:
        """
        Sets the train_param default values (if not defined). log in case of updating a key.
        :param config: train_config
        :param mode: either 'infer' or 'train' (in case of infer, only device and virtual_batch_size are set)
        :return: dict containing all expected values
        """

        def set_default(key, value):
            if key not in full_config:
                full_config[key] = value
                self.logger.info(f"Key {key} not found in config parameter, setting value to default ({value})", {'color': 'yellow'})

        full_config = {} if config is None else config.copy()
        set_default('device', 'cuda')
        set_default('virtual_batch_size', 1)

        if mode == 'train':
            set_default('num_epochs', 100)
            set_default('gap_between_saving_epochs', 5)
            set_default('start_saving_epochs', 80)

        return full_config

    def _handle_dataset_summaries(self, train_dataloader: DataLoader, validation_dataloader: DataLoader) -> None:
        """
        Write summary of datasets into files and logger
        :param train_dataloader: train data
        :param validation_dataloader: validation data (can be None)
        """
        # train dataset summary
        dataset_summary = train_dataloader.dataset.summary()

        train_dataset_summary_file = os.path.join(self.state.output_model_dir, 'train_dataset_summary.txt')
        with open(train_dataset_summary_file, 'w') as sum_file:
            sum_file.write(dataset_summary)
        self.logger.info("Train Dataset Summary:")
        self.logger.info(dataset_summary)

        # validation dataset summary, if exists
        if validation_dataloader is not None:
            dataset_summary = validation_dataloader.dataset.summary()
            validation_dataset_summary_file = os.path.join(self.state.output_model_dir, 'validation_dataset_summary.txt')
            with open(validation_dataset_summary_file, 'w') as sum_file:
                sum_file.write(dataset_summary)
            self.logger.info("Validation Dataset Summary:")
            self.logger.info(dataset_summary)
        pass


def _extend_results_dict(mode: str, current_dict: Dict, aggregated_dict: Dict) -> Dict:
    """
    Utility function to create aggregated loss results dict from the handle_batch()/handle_virtual_batch() output dict.
    :param mode: Can be either 'train', 'validation', 'infer' (on infer return an empty dict)
    :param current_dict: batch_dict or virtual_batch_result dict
    :param aggregated_dict: the aggregated dict to add current_dict
    :return: aggregated dict
    """
    if mode == 'infer':
        return {}
    else:
        # handle the case where batch dict is empty (the end of the last virtual mini batch)
        if current_dict == {}:
            return aggregated_dict
        # for train and validation we need the loss values
        cur_keys = FuseUtilsHierarchicalDict.get_all_keys(current_dict)
        # aggregate just keys that start with losses
        cur_keys = [key for key in cur_keys if key.startswith('losses.')]
        agg_keys = FuseUtilsHierarchicalDict.get_all_keys(aggregated_dict)
        for key in cur_keys:
            if key not in agg_keys:
                # init dict is needed
                FuseUtilsHierarchicalDict.set(aggregated_dict, key, [])
            val = FuseUtilsHierarchicalDict.get(current_dict, key)
            if isinstance(val, list):
                # in the epoch dict, the loss is a list of the virtual mini batches
                FuseUtilsHierarchicalDict.get(aggregated_dict, key).extend(val)
            else:
                # in the virtual batch dict, the losses are float per batch
                FuseUtilsHierarchicalDict.get(aggregated_dict, key).append(val)

    return aggregated_dict
