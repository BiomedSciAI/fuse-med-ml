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
from distutils.dir_util import copy_tree
from shutil import copy
from ehrtransformers.utils.common import create_folder
import torch
import sys
from torch.utils.tensorboard import SummaryWriter
from ehrtransformers.model.model_selector import model_type

base_out_dir_key = 'output_dir_main' #path for all experiments on this type of data and training task
global_base_out_dir_key = 'uber_base_path' #path for all ehr experiments, where global statistics summary is saved


def get_log_dir_name(config, dir_ind):
    return os.path.join(config[base_out_dir_key],'run_{}'.format(dir_ind), 'Logs')

def get_output_dir(config, index=None):
    if index is None:
        curr_run_ind = read_curr_dir_ind(config)
    elif index >0: #index is the absolute index of the output dir
        curr_run_ind = index
    else: #i.e. index<=0 is the relative index of the output dir (0 - last one used, -1 - the one before last, etc.)
        curr_run_ind = read_curr_dir_ind(config) + index #TODO: Some existing directory indices may be missing (e.g. deleted). Instead of just subtracting the relative index, first list all dirs in the location and take the index-before-last one
    return get_log_dir_name(config,curr_run_ind)
    

def get_ind_file_path(config):
    return os.path.join(config[base_out_dir_key], 'curr_ind.txt')

def read_curr_dir_ind(config):
    ind_file_path = get_ind_file_path(config)
    if os.path.exists(ind_file_path):
        with open(ind_file_path, 'r') as f:
            curr_ind = int(f.readline())
    else:
        curr_ind = 100
    return curr_ind

def get_and_advance_dir_ind(config):
    """
    Reads and advances an output dir index
    :param config:
    :return:
    """
    if not os.path.exists(config[base_out_dir_key]):
        os.makedirs(config[base_out_dir_key])

    ind_file_path = get_ind_file_path(config)
    curr_ind = read_curr_dir_ind(config)+1

    with open(ind_file_path, 'w') as f:
        f.write(str(curr_ind))

    return curr_ind

def make_new_out_dir(config):
    """
    Creates an output directory with a new index
    :param config:
    :return:
    """
    dir_ind = get_and_advance_dir_ind(config)
    path = os.path.join(config[base_out_dir_key], 'run_{}'.format(dir_ind))
    if os.path.exists(path):
        if len(os.listdir(path)) > 2:
            raise Exception("log output path already exists")
    else:
        os.makedirs(path)
    return path, dir_ind


def copy_source(out_main_path):
    """
    Creates an output directory with a new index (advancing output index in the process) and copies relevant configs and scripts there.
    :param config:
    :return:
    """
    subdir_list = ['.']  #['ehrtransformers', 'scripts']

    log_path = os.path.abspath(__file__)
    code_path = os.path.dirname(os.path.dirname(os.path.dirname(log_path)))
    # BEHRT_path = os.path.join(os.path.join(log_path, os.pardir), os.pardir)

    out_code_path = os.path.join(out_main_path, 'Code')
    out_log_path = os.path.join(out_main_path, 'Logs')

    os.makedirs(out_log_path, exist_ok=True)

    #copy all relevant run files to the log dir:
    for subdir in subdir_list:
        tmp_path_out = os.path.join(out_code_path, subdir)
        tmp_path_in = os.path.join(code_path, subdir)
        os.makedirs(tmp_path_out, exist_ok=True)
        copy_tree(tmp_path_in, tmp_path_out)

    return out_log_path

def get_run_index_from_model_location(model_path):
    dirs = model_path.split('/')
    run_dirs = [d for d in dirs if d.startswith('run_')]
    if len(run_dirs) < 1:
        raise Exception('could not extract run index from model path')
    run_ind = int(run_dirs[-1].split('_')[-1])
    return run_ind

class Logger():
    file_config=None
    model_config = None
    data_config = None
    global_params = None
    glob_summary_filename = None #filename of the global summary file
    stdout_save = None
    stderr_save = None
    local_log_file = None #log file to dump all printouts of the current run.
    summary_writer = None
    out_main_path = None
    def __init__(self, file_config, model_config, data_config, global_params, main_argv = None, index = None, override_output=True):
        """
        Creates a new logger instance
        :param file_config:
        :param model_config:
        :param data_config:
        :param global_params:
        :param index: If None, a new output dir is created, and its index advanced.
            If an integer >0, then that is used as index. 0 means use the latest output dir, -1 - previous one and so on.
            TODO: So far - only None index was implemented
        :param override_output: If True redirects stdout to a file
        """
        self.do_debug = global_params['debug_mode']
        self.file_config = file_config
        self.model_config = model_config
        self.data_config = data_config
        self.global_params = global_params
        self.glob_summary_filename = os.path.join(global_params[global_base_out_dir_key], global_params['global_stat_file'])
        self.main_argv = main_argv
        self.output_index = index
        
        if index is None:
            # Now, this is tricky. The latest output dir index is stored in a text file (curr_ind.txt) in the output dir.
            # Every time we create a logger instance with index=None, it advances the index and creates a new output dir.
            # However, between updating curr_ind.txt and reading the index from there, another run may update it, resulting
            # in the wrong index being read. In order to avoid this we should probably protect curr_ind.txt with a mutex (TO)
            # but in the meantime we need to make sure it's only accessed ONCE for every instance of Logger. Since we access
            # curr_ind.txt in make_new_out_dir and in get_output_dir, I made sure that the latter accesses it only if self.output_index=None
            # (which is not the case if self.output_index was previously set by make_new_out_dir), and a Logger function
            # that reads the index returns a local copy only.
            self.out_main_path, self.output_index = make_new_out_dir(global_params) # creates an output dir and advances its index
            copy_source(self.out_main_path)  # copies relevant sources
        self.global_params["output_dir"] = get_output_dir(global_params, self.output_index)

        create_folder(global_params['output_dir'])

        # override stdout and stderr with a log file
        self.stdout_save = sys.stdout
        self.stderr_save = sys.stderr

        gettrace = getattr(sys, 'gettrace', None)

        if override_output: #not self.do_debug: #gettrace is None: #True: # Identify if we're in debug mode and override stdout if we're not
            print('No sys.gettrace') #Seems like we're not in debug mode
            self.local_log_file = open(os.path.join(global_params['output_dir'], 'log.txt'), 'w')
            sys.stdout = self.local_log_file
            sys.stderr = self.local_log_file
        else:
            print('In debug mode')


        if (self.main_argv is not None) and len(self.main_argv) > 1:
            config_file_path = self.main_argv[1]
            print('config file path: {}'.format(config_file_path))

        if data_config['days_to_inddate_tr'] == None:
            train_visit_str = 'all visits'
        else:
            if data_config['days_to_inddate_start_tr'] == None:
                train_visit_str = f'start..{-data_config["days_to_inddate_tr"]}'
            else:
                train_visit_str = f'{-data_config["days_to_inddate_start_tr"]}..{-data_config["days_to_inddate_tr"]}'
        if data_config['days_to_inddate'] == None:
            test_visit_str = 'all visits'
        else:
            if data_config['days_to_inddate_start'] == None:
                test_visit_str = f'start..{-data_config["days_to_inddate"]}'
            else:
                test_visit_str = f'{-data_config["days_to_inddate_start"]}..{-data_config["days_to_inddate"]}'

        self.log_ID_string = f'run_{self.output_index}: model {model_type} train on {train_visit_str}, test on {test_visit_str} days after index, batch: {global_params["batch_size"]}, layers: {model_config["num_hidden_layers"]}, att heads: {model_config["num_attention_heads"]}, embedding: {model_config["hidden_size"]}  , reverse_input: {model_config["reverse_input_direction"]}, procedure: {data_config["use_procedures"]}, {model_config["heads"]} weights= {model_config["head_weights"]}, visit_day_resolution: {data_config["visit_days_resolution"]}'

        print(self.log_ID_string)
        # tensorboard:
        tb_comment = 'disease: {}, outcome: {}, repr_length: {}, days_to_ind: {}, MB_size: {}'.format(
                                                            self.data_config['task'],
                                                            self.data_config['task_type'],
                                                            self.model_config['hidden_size'],
                                                            self.data_config['days_to_inddate'],
                                                            self.global_params['batch_size'])
        # self.summary_writer = SummaryWriter(log_dir=os.path.join(global_params['output_dir'], 'TB'),comment=tb_comment)
    def __str__(self):
        return self.log_ID_string

    def __del__(self):
        # close log file and restore stdout and stderr
        sys.stdout = self.stdout_save
        sys.stderr = self.stderr_save
        if self.local_log_file is not None:
            self.local_log_file.close()

    def get_out_dir_index(self):
        return self.output_index

    def get_out_dir_path(self):
        return self.out_main_path


    def save_model(self, model_to_save, prefix=""):
        """
        Saves model if needed (i.e. if global_params["save_model"] is true.
        :param model_to_save:
        :param prefix:
        :return:
        """
        output_model_file = os.path.join(self.global_params['output_dir'], prefix+(self.global_params['best_name']))
        # create_folder(global_params['output_dir'])
        if self.global_params['save_model']:
            torch.save(model_to_save, output_model_file)

    # def add_local_summary_scalars(self, iter_n, scal_dict, mode='val'):
    #     for k in scal_dict:
    #        self.summary_writer.add_scalar('{}/{}'.format(k,mode), scal_dict[k], iter_n)

    def add_summary_line(self, last_AUC, last_avgprecision, last_epoch, best_AUC, best_avgprecision, best_epoch):
        add_titles = False
        self.glob_summary_filename
        titles = '{}~{}~{}~{}~{}~{}~{}~{}~{}~{}~{}~{}~{}~{}~{}~{}\n'.format('disease', 'outcome', 'repr_length', 'days_to_ind', 'MB_size', 'POS_patients', 'NEG_patients', 'POS_visits', 'NEG_visits', 'AUC_last', 'AUC_best', 'iter_last', 'iter_best', 'PREC_last', 'PREC_best', 'log_path', )
        line =   '{}~{}~{}~{}~{}~{}~{}~{}~{}~{}~{}~{}~{}~{}~{}~{}\n'.format(  self.data_config['task'],
                                                            self.data_config['task_type'],
                                                            self.model_config['hidden_size'],
                                                            self.data_config['days_to_inddate'],
                                                            self.global_params['batch_size'],
                                                            -1,
                                                            -1,
                                                            -1,
                                                            -1,
                                                            last_AUC,
                                                            best_AUC,
                                                            last_epoch,
                                                            best_epoch,
                                                            last_avgprecision,
                                                            best_avgprecision,
                                                            self.global_params['output_dir'])
        if ~os.path.exists(self.glob_summary_filename):
            add_titles = True
        with open(self.glob_summary_filename, 'a') as file_obj:
            if add_titles:
                file_obj.write(titles)
            file_obj.write(line)
            file_obj.flush()

    def save_file(self, file_location, output_subdir):
        if self.out_main_path is None:
            raise RuntimeError("No output path defined")
        out_path = os.path.join(self.out_main_path, output_subdir)
        copy(src=file_location, dst=out_path)
