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

import pandas as pd
from typing import Sequence, Hashable, Union, Optional, List, Dict

from fuse.data.data_source.data_source_base import FuseDataSourceBase
from fuse.utils.utils_misc import autodetect_input_source


class FuseDataSourceDefault(FuseDataSourceBase):
    """
    DataSource for the following aut-detectable types:

    1. DataFrame (instance or path to pickled object)
    2. Python list of sample descriptors
    3. Text file (needs to end with '.txt' or '.text' extension)

    """

    def __init__(self, input_source: Union[str, pd.DataFrame, Sequence[Hashable]] = None,
                 folds: Optional[Union[int, Sequence[int]]] = None, conditions: Optional[List[Dict[str, List]]] = None) -> None:
        """
        :param input_source: auto-detectable input source
        :param folds:   if input is a DataFrame having a 'fold' column, filter by this fold(s)
        :param conditions: conditions to apply on data source.
                the conditions are column names that are expected to be in input_source data frame.

                Structure:
                   * List of 'Filter Queries' with logical OR between them.
                   * Each Filter Query is a dictionary of data source column and a list of possible values, with logical AND between the keys.

                Example - selecting only negative or positive biopsy samples:
                   [{'biopsy' : ['positive', 'negative']}]
                Example - selecting negative or positive biopsy biopsy samples that are of type 'tumor':
                   [{'biopsy': ['positive', 'negative'], 'type': ['tumor']}]
                Example - selecting negative/positive biopsy samples that are of type 'calcification' AND marked as BIRAD 0 or 5:
                   [{'biopsy': ['positive', 'negative'], 'type': ['calcification'], 'birad': ['BIRAD0', 'BIRAD5']}]
                Example - selecting samples that are either positive biopsy OR marked as BIRAD 0:
                   [{'biopsy': ['positive']}, {'birad': ['BIRAD0']}]

        """
        self.samples_df = autodetect_input_source(input_source)

        if conditions is not None:
            before = len(self.samples_df)
            to_keep = self.filter_by_conditions(self.samples_df, conditions)
            self.samples_df = self.samples_df[to_keep].copy()
            logging.getLogger('Fuse').info(f"Remove {before - len(self.samples_df)} records that did not meet conditions")

        if self.samples_df is None:
            raise Exception('Error detecting input source in FuseDataSourceDefault')

        if isinstance(folds, int):
            self.folds = [folds]
        else:
            self.folds = folds

        if self.folds is not None:
            assert 'fold' in self.samples_df, f'Data cannot be filtered by folds {folds} as folds are specified in the collected data'
            self.samples_df = self.samples_df[self.samples_df['fold'].isin(self.folds)]

    @staticmethod
    def filter_by_conditions(samples: pd.DataFrame, conditions: Optional[List[Dict[str, List]]]):
        """
        Returns a vector of the samples that passed the conditions
        :param samples: dataframe to check. expected to have at least sample_desc column.
        :param conditions: list of dictionaries. each dictionary has column name as keys and possible values as the values.
                for each dict in the list:
                    the keys are applied with AND between them.
                the dict conditions are applied with OR between them.
        :return: boolean vector with the filtered samples
        """
        to_keep = samples.sample_desc.isna()  # start with all false
        for condition_list in conditions:
            condition_to_keep = samples.sample_desc.notna()  # start with all true
            for column, values in condition_list.items():
                condition_to_keep = condition_to_keep & samples[column].isin(values)  # all conditions in list must be met
            to_keep = to_keep | condition_to_keep  # add this condition samples to_keep
        return to_keep

    def get_samples_description(self):
        return list(self.samples_df['sample_desc'])

    def summary(self) -> str:
        summary_str = ''
        summary_str += 'FuseDataSourceDefault - %d samples\n' % len(self.samples_df)
        return summary_str


if __name__ == '__main__':
    my_df = pd.DataFrame({'sample_desc': range(11, 16),
                          'A': range(1, 6),
                          'B': range(10, 0, -2),
                          'C': range(10, 5, -1)})
    print(my_df)
    clist = [{'A': [2, 3, 4], 'B': [8, 2]}, {'C': [8, 7]}]
    to_keep = FuseDataSourceDefault.filter_by_conditions(my_df, clist)
    print(my_df[to_keep])

    to_keep = FuseDataSourceDefault.filter_by_conditions(my_df, [{}])
    print(my_df[to_keep])
