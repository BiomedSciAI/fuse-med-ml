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

import inspect
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from multiprocessing import current_process
from shutil import copyfile
from typing import Dict, Optional, Any, List

from termcolor import colored

from fuse.utils.utils_file import FuseUtilsFile


class ProcessSafeHandler(logging.StreamHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._locks = defaultdict(lambda: threading.RLock())

    def acquire(self):
        current_process_id = current_process().pid
        self._locks[current_process_id].acquire()

    def release(self):
        current_process_id = current_process().pid
        self._locks[current_process_id].release()


class FuseConsoleFormatter(logging.Formatter):
    """Logging Formatter to add colors per verbose level and file, line number in case of warnning/error"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    warn_format = "(%(filename)s:%(lineno)d) WARNING: %(message)s"
    err_format = "(%(filename)s:%(lineno)d) ERROR: %(message)s"

    LEVEL_FORMATS = {
        logging.WARNING: yellow + warn_format + reset,
        logging.ERROR: red + err_format + reset,
        logging.CRITICAL: bold_red + err_format + reset
    }

    def format(self, record):
        if record.levelno in [logging.INFO, logging.DEBUG]:
            if isinstance(record.__dict__['args'], dict):
                color = record.__dict__['args'].get('color', None)
                on_color = record.__dict__['args'].get('on_color', None)
                attrs = record.__dict__['args'].get('attrs', None)
                if attrs is not None and not isinstance(attrs, list):
                    attrs = [attrs]
                log_fmt = colored('%(message)s', color=color,
                                  on_color=on_color,
                                  attrs=attrs)
            else:
                log_fmt = '%(message)s'
        else:
            log_fmt = self.LEVEL_FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)

        return formatter.format(record)


def fuse_logger_start(output_path: Optional[str] = None, console_verbose_level: int = logging.INFO,
                      list_of_source_files: Optional[List[str]] = None) -> None:
    """
    Start Fuse logger:
    Define output destination; console, file and file including also debug
    Define the formats:
    - Console - color per verbose level and (file:line) prefix in a case of warning or error
    - File - adds prefix '(file:line) : level'
    - Debug file - adds prefix 'time: (file:line) : level'

    :param output_path: place to save the log files. If not specified log to files will be disabled
    :param console_verbose_level: verbose level used to print to the screen
    :param force_reset: if logs already exist in output_path - delete automatically (True) or prompt user before deletion, default is False
    :param list_of_source_files: list of source files or paths to save (relative path to the working directory)
           If None, just the file of the caller fuction will be saved

    :return: None
    """
    lgr = logging.getLogger('Fuse')
    # skip if already configured
    if lgr.level != 0:
        return
    lgr.setLevel(logging.DEBUG)

    # console
    console_handler = ProcessSafeHandler(stream=sys.stdout)
    console_handler.setLevel(console_verbose_level)
    console_formatter = FuseConsoleFormatter()
    console_handler.setFormatter(console_formatter)
    lgr.addHandler(console_handler)
    lgr.propagate = False

    # time str to be used by filename
    timestr = time.strftime("%d_%m_%y__%H_%M")

    # log to files - only if output_path is provided
    if output_path is not None:
        log_output_path = os.path.join(output_path, 'logs')
        # Create log dir
        FuseUtilsFile.create_dir(log_output_path)

        # file info
        file_handler = logging.FileHandler(os.path.join(log_output_path, f'fuse_{timestr}.log'))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('(%(filename)s:%(lineno)d) : %(levelname)s : %(message)s')
        file_handler.setFormatter(formatter)
        lgr.addHandler(file_handler)

        # file verbose
        file_handler = RotatingFileHandler(os.path.join(log_output_path, 'fuse_verbose.log'), maxBytes=1e6)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s : (%(filename)s:%(lineno)d) : %(message)s')
        file_handler.setFormatter(formatter)
        lgr.addHandler(file_handler)

    start_time = time.strftime("%d/%m/%y %H:%M:%S")
    lgr.info(f'Fuse: {start_time}', {'color': 'cyan', 'attrs': ['dark', 'bold']})

    if list_of_source_files is None or list_of_source_files:
        if output_path is not None:
            source_files_output_path = os.path.join(output_path, f'source_files', timestr)
            FuseUtilsFile.create_or_reset_dir(source_files_output_path)

            lgr.info(f'Copy source files to {source_files_output_path}')
            if list_of_source_files is None:
                # copy just the caller function file name
                caller_function_file_name = inspect.stack()[1][1]
                list_of_source_files = [caller_function_file_name]

            for src_file in list_of_source_files:
                copyfile(os.path.abspath(src_file), os.path.join(source_files_output_path, os.path.basename(src_file)))


def convert_state_to_str(input_state: Any):
    """
    Convert a state built from a nested dictionaries, lists and tuples to a string
    :param input_state:
    :return:
    """
    if isinstance(input_state, dict):
        return {param_name: convert_state_to_str(input_state[param_name]) for param_name in input_state}
    if isinstance(input_state, list):
        return [convert_state_to_str(param) for param in input_state]
    if isinstance(input_state, tuple):
        return tuple((convert_state_to_str(param) for param in input_state))
    return str(input_state)


def log_object_input_state(obj: Any, input_state: Dict) -> None:
    """
    Log object creation: module name, address and input parameters
    :param obj: The object, typically set to 'self'.
    :param input_state: The input parameters,
                             To get them call this function first thing in the constructors with input_parameters=locals()
    :return: None
    """
    lgr = logging.getLogger('Fuse')
    input_state_copy = input_state.copy()
    # remove self from the state if exist
    input_state_copy.pop('self', None)
    input_paramters_str = convert_state_to_str(input_state_copy)
    lgr.debug(f'\n-----\n{str(obj)} created: \n\tinput state {input_paramters_str} \n-----\n')
