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

Created on January 06, 2022

"""

import pandas as pd
import os
import numpy as np
from fuse.data.data_source.data_source_base import FuseDataSourceBase
from typing import Optional, Tuple
from fuse.data.data_source.data_source_toolbox import FuseDataSourceToolbox


class FuseDataSourceFolds(FuseDataSourceBase):
    def __init__(self,
                 input_source: str,
                 input_df : pd.DataFrame,
                 phase: str,
                 no_mixture_id: str,
                 balance_keys: np.ndarray,
                 reset_partition_file: bool,
                 folds: Tuple[int],
                 num_folds : int =5,
                 partition_file_name: str = None
                 ):

        """
        Create DataSource which is divided to num_folds folds, supports either a path to a csv or data frame as input source.
        The function creates a partition file which saves the fold partition
        :param input_source:       path to dataframe containing the samples ( optional )
        :param input_df:           dataframe containing the samples ( optional )
        :param no_mixture_id:      The key column for which no mixture between folds should be forced
        :param balance_keys:       keys for which balancing is forced
        :param reset_partition_file: boolean flag which indicate if we want to reset the partition file
        :param folds               indicates which folds we want to retrieve from the fold partition
        :param num_folds:          number of folds to divide the data
        :param partition_file_name:name of a csv file for the fold partition
                                   If train = True, train/val indices are dumped into the file,
                                   If train = False, train/val indices are loaded
        :param phase:              specifies if we are in train/validation/test/all phase
        """
        self.nfolds = num_folds
        self.key_columns = balance_keys
        if reset_partition_file is True and phase not in ['train','all']:
            raise Exception("Sorry, it is possible to reset partition file only in train / all phase")
        if reset_partition_file is True or not os.path.isfile(partition_file_name):
            # Load csv file
             # ----------------------

             if input_source is not None :
                input_df = pd.read_csv(input_source)
                self.folds_df = FuseDataSourceToolbox.balanced_division(df = input_df ,
                                                                    no_mixture_id = no_mixture_id,
                                                                    key_columns = self.key_columns ,
                                                                    nfolds = self.nfolds ,
                                                                    print_flag=True )
             # Extract entities
             # ----------------
        else:
             self.folds_df = pd.read_csv(partition_file_name)

        sample_descs = []
        for fold in folds:
            sample_descs += self.folds_df[self.folds_df['fold'] == fold]['file'].to_list()

        self.samples = sample_descs

        self.input_source = input_source

    def get_samples_description(self):
        """
        Returns a list of samples ids.
        :return: list[str]
        """
        return self.samples

    def summary(self) -> str:
        """
        Returns a data summary.
        :return: str
        """
        summary_str = ''
        summary_str += 'Class = '+type(self).__name__+'\n'

        if isinstance(self.input_source, str):
            summary_str += 'Input source filename = %s\n' % self.input_source

        summary_str += FuseDataSourceToolbox.print_folds_stat(db = self.folds_df ,
                                                                nfolds = self.nfolds ,
                                                                key_columns = self.key_columns )

        return summary_str
