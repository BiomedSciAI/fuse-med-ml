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

from typing import Any

from fuse.utils.utils_misc import Singleton


class FuseUtilsDebug(metaclass=Singleton):
    """
    Debug settings. See __init__() for available modes
    """

    def __init__(self, mode: str = 'default'):
        """
        :param mode: debug mode that can later be customize by override_setting().
                    Supported modes:
                    - 'default' - no debug tools
                    - 'debug' - best settings to use debugger: disable multiprocessing and multi threading
                    - 'verbose' - print the the data collected in batch_dict and epoch_results in every stage with debug flag enabled.
                    - 'fast' - run single batch and epoch with verbose and debug flag enabled
                    - 'user' - prints when user func begin and ends
        """
        # possible values for each attribute
        self._settings_supported_values = {
            # display the info as provided to Fuse callbacks
            'dataset_sample_stages_info': ['default', 'verbose'],
            # allows to override number of samples in dataset
            'dataset_override_num_samples': lambda x: isinstance(x, int) and x > 0,
            # allows to override number of cache workers
            'dataset_override_num_workers': lambda x: isinstance(x, int) and x >= 0,
            # 'verbose': display each time user function begin and ends
            'dataset_user': ['default', 'verbose'],
            # get random sample from available classes
            'sampler_batch_mode': ['default', 'simple'],
            # allows to override number of dataloader data workers
            'manager_override_num_dataloader_workers': lambda x: isinstance(x, int) and x >= 0,
            # allows to override number of epochs
            'manager_override_num_epochs': lambda x: isinstance(x, int) and x > 0,
            # allows to override number of gpus
            'manager_override_num_gpus': lambda x: isinstance(x, int) and x >= 0,
            # 'verbose': display each stage
            'manager_stages': ['default', 'verbose'],
            # 'verbose': display each time user function begin and ends
            'manager_user': ['default', 'verbose'],
        }
        # defined the supported modes
        self._modes = {}
        # normal mode
        default_settings = {key: 'default' for key in self._settings_supported_values}
        self._modes['default'] = default_settings
        # debug mode
        debug_settings = {
            'dataset_override_num_workers': 0,
            'manager_override_num_gpus': 1,
            'manager_override_num_dataloader_workers': 0
        }
        self._modes['debug'] = debug_settings
        # verbose mode
        verbose_settings = {
            # 'dataset_info': 'verbose',
            'dataset_sample_stages_info': 'verbose',
        }
        verbose_settings.update(debug_settings)
        self._modes['verbose'] = verbose_settings
        # fast mode
        fast_settings = {
            'dataset_override_num_samples': 10,
            'sampler_batch_mode': 'simple',
            'manager_override_num_epochs': 2
        }
        fast_settings.update(verbose_settings)
        self._modes['fast'] = fast_settings
        # user mode
        user_settings = {
            # FIXME: Implement
            # 'manager_user': 'verbose',
            # 'dataset_user': 'verbose'
        }
        self._modes['user'] = user_settings

        self._settings: dict

        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        """
        set debug mode
        :param mode: see __init__{} for available modes
        """
        self._settings = self._modes['default']
        self.override_mode(mode)

    def override_mode(self, mode: str) -> None:
        """
        Override just the settings relevant to this mode
        :param mode: see __init__{} for available modes
        """
        assert mode in self._modes
        if mode != 'default':
            for key, value in self._modes[mode].items():
                self.set_setting(key, value)

    def set_setting(self, name: str, value: Any) -> None:
        """
        :param name: setting name. See self._settings_supported_values for available settings.
        :param value: value to override to. See self._settings_supported_values for possible values.
        """
        assert name in self._settings_supported_values, f'setting {name} is not supported'
        supported_values = self._settings_supported_values[name]
        if isinstance(supported_values, list):
            assert value in supported_values, f'value {value} is not supported for setting {name}, supported values are {supported_values}'
        else:
            assert supported_values(value), f'value {value} is not supported for setting {name}'

        self._settings[name] = value

    def get_setting(self, name: str) -> Any:
        """
        :param name: setting name. See self._settings_supported_values for possible settings.
        :return: te value of that setting. See self._settings_supported_values for possible values.
        """
        assert name in self._settings_supported_values, f'setting {name} is not supported'
        return self._settings[name]
