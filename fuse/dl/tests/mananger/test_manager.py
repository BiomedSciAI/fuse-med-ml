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

from ntpath import join
from tempfile import tempdir
import tempfile
import unittest
import os
import logging
from fuse.utils.utils_logger import fuse_logger_start

from fuse.dl.managers.manager_default import FuseManagerDefault
from fuse.utils.file_io.file_io import create_or_reset_dir


class FuseManagerTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        create_or_reset_dir(self.tempdir, force_reset=True)
        fuse_logger_start(output_path=self.tempdir, console_verbose_level=logging.INFO)

        self.manager = FuseManagerDefault(self.tempdir, force_reset=True)
        self.train_dict = {'metric_1': 100, 'metric_2': 80, 'metric_3': 75}
        self.validation_dict = {'metric_1': 90, 'metric_2': 70, 'metric_3': 60}
        self.manager.state.current_epoch = 7
        pass

    def read_file(self):
        with open(os.path.join(self.tempdir, "last_epoch_summary.txt"), 'r') as sfile:
            summ = sfile.read()
        return summ

    def test_epoch_summary_one_source(self):
        self.manager.state.best_epoch_function = ['metric_1']
        self.manager.state.best_epoch = [5]
        self.manager.state.best_epoch_values = [{'metric_1': 85, 'metric_2': 84, 'metric_3': 71}]
        self.manager._write_epoch_summary_table(self.train_dict, self.validation_dict, 0)

        summary = self.read_file()
        self.assertTrue('| metric_1                 | 85.0000                  | 90.0000                  | 100.0000                 |' in summary)
        self.assertTrue('| metric_2                 | 84.0000                  | 70.0000                  | 80.0000                  |' in summary)
        self.assertTrue('| metric_3                 | 71.0000                  | 60.0000                  | 75.0000                  |' in summary)
        self.assertTrue('Stats for epoch: 7 (Best epoch is 5 for source metric_1)' in summary)

    def test_epoch_summary_two_sources(self):
        self.manager.state.best_epoch_function = ['metric_1', 'metric_2']
        self.manager.state.best_epoch = [5, 12]
        self.manager.state.best_epoch_values = [{'metric_1': 85, 'metric_2': 84, 'metric_3': 71}, {'metric_1': 81, 'metric_2': 89, 'metric_3': 65}]
        self.manager._write_epoch_summary_table(self.train_dict, self.validation_dict, 0)

        summary = self.read_file()
        self.assertTrue('| metric_1                 | 85.0000                  | 90.0000                  | 100.0000                 |' in summary)
        self.assertTrue('| metric_2                 | 84.0000                  | 70.0000                  | 80.0000                  |' in summary)
        self.assertTrue('| metric_3                 | 71.0000                  | 60.0000                  | 75.0000                  |' in summary)
        self.assertFalse('| metric_3                 | 65.0000                  | 60.0000                  | 75.0000                  |' in summary)
        self.assertTrue('Stats for epoch: 7 (Best epoch is 5 for source metric_1)' in summary)

        self.manager._write_epoch_summary_table(self.train_dict, self.validation_dict, 1)
        summary = self.read_file()
        # see that index 0 source was not deleted
        self.assertTrue('| metric_1                 | 85.0000                  | 90.0000                  | 100.0000                 |' in summary)
        self.assertTrue('| metric_2                 | 84.0000                  | 70.0000                  | 80.0000                  |' in summary)
        self.assertTrue('| metric_3                 | 71.0000                  | 60.0000                  | 75.0000                  |' in summary)
        # see that index 1 was added
        self.assertTrue('| metric_1                 | 81.0000                  | 90.0000                  | 100.0000                 |' in summary)
        self.assertTrue('| metric_2                 | 89.0000                  | 70.0000                  | 80.0000                  |' in summary)
        self.assertTrue('| metric_3                 | 65.0000                  | 60.0000                  | 75.0000                  |' in summary)
        self.assertTrue('Stats for epoch: 7 (Best epoch is 12 for source metric_2)' in summary)

    def test_epoch_summary_none_values(self):
        self.manager.state.best_epoch_function = ['metric_1']
        self.manager.state.best_epoch = [5]
        train_dict = {'metric_1': 100, 'metric_2': 80, 'metric_3': None}
        validation_dict = {'metric_1': 90, 'metric_2': None, 'metric_3': None}
        self.manager.state.best_epoch_values = [{'metric_1': None, 'metric_2': 84, 'metric_3': None}]
        self.manager._write_epoch_summary_table(train_dict, validation_dict, 0)

        summary = self.read_file()
        self.assertTrue('| metric_1                 | N/A                      | 90.0000                  | 100.0000                 |' in summary)
        self.assertTrue('| metric_2                 | 84.0000                  | N/A                      | 80.0000                  |' in summary)
        self.assertTrue('| metric_3                 | N/A                      | N/A                      | N/A                      |' in summary)
        self.assertTrue('Stats for epoch: 7 (Best epoch is 5 for source metric_1)' in summary)

    def test_epoch_summary_string_values(self):
        self.manager.state.best_epoch_function = ['metric_1']
        self.manager.state.best_epoch = [5]
        train_dict = {'metric_1': 100, 'metric_2': 80, 'metric_3': 'lala'}
        validation_dict = {'metric_1': 90, 'metric_2': 'lili', 'metric_3': 'lolo'}
        self.manager.state.best_epoch_values = [{'metric_1': 14, 'metric_2': 84, 'metric_3': 'kiki'}]
        self.manager._write_epoch_summary_table(train_dict, validation_dict, 0)

        summary = self.read_file()
        self.assertTrue('| metric_1                 | 14.0000                  | 90.0000                  | 100.0000                 |' in summary)
        self.assertTrue('| metric_2                 | 84.0000                  | N/A                      | 80.0000                  |' in summary)
        self.assertTrue('| metric_3                 | N/A                      | N/A                      | N/A                      |' in summary)
        self.assertTrue('Stats for epoch: 7 (Best epoch is 5 for source metric_1)' in summary)

    def test_is_best_epoch_so_far(self):
        print("test_is_best_epoch_so_far")

        self.manager.state.on_equal_values = ['better', 'worse']
        self.manager.state.optimization_function = ['max', 'max']
        self.manager.state.best_epoch_function = ['metric_1', 'metric_2']
        self.manager.state.best_epoch_values = [{'metric_1': 85, 'metric_2': 84, 'metric_3': 71},
                                          {'metric_1': 81, 'metric_2': 89, 'metric_3': 65}]
        train_dict = {}
        validation_dict = {'metric_1': 90, 'metric_2': None, 'metric_3': 'lolo'}

        is_better = self.manager._is_best_epoch_so_far(train_dict, validation_dict, 0)
        self.assertTrue(is_better)
        self.assertDictEqual(self.manager.state.best_epoch_values[0], validation_dict)

        is_better = self.manager._is_best_epoch_so_far(train_dict, validation_dict, 1)
        self.assertFalse(is_better)
        self.assertDictEqual(self.manager.state.best_epoch_values[0], {'metric_1': 90, 'metric_2': None, 'metric_3': 'lolo'})

        validation_dict = {'metric_1': 90, 'metric_2': 89, 'metric_3': 'lolo'}
        is_better = self.manager._is_best_epoch_so_far(train_dict, validation_dict, 1)
        self.assertFalse(is_better)
        self.assertDictEqual(self.manager.state.best_epoch_values[1], {'metric_1': 81, 'metric_2': 89, 'metric_3': 65})

        validation_dict = {'metric_1': 90, 'metric_2': 89.0001, 'metric_3': 'lolo'}
        is_better = self.manager._is_best_epoch_so_far(train_dict, validation_dict, 1)
        self.assertTrue(is_better)
        self.assertDictEqual(self.manager.state.best_epoch_values[1], validation_dict)

    def test_is_best_epoch_so_far_invalid_metric(self):
        print("test_is_best_epoch_so_far")

        self.manager.state.on_equal_values = ['better', 'worse']
        self.manager.state.optimization_function = ['max', 'max']
        self.manager.state.best_epoch_function = ['wrong_metric', 'metric_2']
        self.manager.state.best_epoch_values = [{'metric_1': 85, 'metric_2': 84, 'metric_3': 71},
                                          {'metric_1': 81, 'metric_2': 89, 'metric_3': 65}]
        train_dict = {}
        validation_dict = {'metric_1': 90, 'metric_2': 95, 'metric_3': 'lolo'}

        with self.assertRaises(KeyError):
            self.manager._is_best_epoch_so_far(train_dict, validation_dict, 0)

        is_better = self.manager._is_best_epoch_so_far(train_dict, validation_dict, 1)
        self.assertTrue(is_better)
        self.assertDictEqual(self.manager.state.best_epoch_values[1], validation_dict)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
