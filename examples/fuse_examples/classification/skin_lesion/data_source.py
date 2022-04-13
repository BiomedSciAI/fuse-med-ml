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

from typing import Optional

import pandas as pd
import numpy as np
import pickle

from fuse.data.data_source.data_source_base import FuseDataSourceBase


class FuseSkinDataSource(FuseDataSourceBase):
    def __init__(self,
                 input_source: str,
                 partition_file: Optional[str] = None,
                 train: bool = True,
                 portion_train: float = 0.7,
                 override_partition: bool = True
                 ):

        """
        Create DataSource
        :param input_source:       path to images
        :param partition_file:     Optional, name of a pickle file when no validation set is available
                                   If train = True, train/val indices are dumped into the file,
                                   If train = False, train/val indices are loaded
        :param train:              specifies if we are in training phase
        :param portion_train:      train proportion in case of splitting
        :param override_partition: specifies if the given partition file is filled with new train/val splits
        """

        # Load csv file
        # ----------------------
        input_df = pd.read_csv(input_source)

        # Extract entities
        # ----------------

        sample_descs = []

        if partition_file is not None:
            if train:
                if override_partition:
                    num_sequences = input_df.shape[0]
                    break_train = int(num_sequences * portion_train)
                    splits = np.random.permutation(np.arange(num_sequences))
                    splits = np.split(splits, [break_train])
                    splits = {'train': splits[0], 'val': splits[1]}

                    with open(partition_file, "wb") as pickle_out:
                        pickle.dump(splits, pickle_out)

                    for sample_id in input_df.iloc[splits['train'], 0]:
                        sample_descs.append(sample_id)
                else:
                    # read from a previous train/test split to evaluate on the same partition
                    with open(partition_file, "rb") as splits:
                        repartition = pickle.load(splits)
                        for sample_id in input_df.iloc[repartition['train'], 0]:
                            sample_descs.append(sample_id)

            else:
                with open(partition_file, "rb") as splits:
                    repartition = pickle.load(splits)
                    for sample_id in input_df.iloc[repartition['val'], 0]:
                        sample_descs.append(sample_id)

        else:
            for sample_id in input_df.iloc[:, 0]:
                sample_descs.append(sample_id)

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
        summary_str += 'Class = FuseSkinDataSource\n'

        if isinstance(self.input_source, str):
            summary_str += 'Input source filename = %s\n' % self.input_source

        summary_str += 'Number of samples = %d\n' % len(self.samples)

        return summary_str
