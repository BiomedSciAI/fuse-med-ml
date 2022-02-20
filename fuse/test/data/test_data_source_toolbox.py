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

import unittest
import pandas as pd
import numpy as np
import torch
import os
from scipy import stats 

from fuse.data.data_source.data_source_toolbox import FuseDataSourceToolbox
import pathlib


class FuseDataSourceToolBoxTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        pass

    def test_balanced_division(self):
        input_df = pd.read_csv(os.path.join(pathlib.Path(__file__).parent.resolve(),'file_for_test.csv'))
        
        # configure input for fold partition
        unique_ID = 'ID1'
        label = 'label1'
        folds = 5
         
        partition_df = FuseDataSourceToolbox.balanced_division(df = input_df ,
                                                                no_mixture_id = unique_ID,
                                                                key_columns =[label] ,
                                                                nfolds = folds ,
                                                                print_flag=False,
                                                                debug_mode = True)
        # set expected values for the partition
        
        # expeted id_level values and unique records
        id_level_value_counter= {'[ True False]': 465, '[False  True]': 1280, '[ True  True]': 30}
        
        # label balance value 
        population_mean = input_df[label].mean()
        
        # observed label balance in folds
        means = [partition_df[partition_df['fold'] == i][label].mean()  for i in range(folds)]
        
        # get number of unique ID in each folds   
        folds_size = [len(partition_df[partition_df['fold'] == i][unique_ID].unique())  for i in range(folds)] 
        
        # number of records in each fold
        records_size = [len(partition_df[partition_df['fold'] == i])  for i in range(folds)]
        
        # confidence level for confidence intervals      
        confidence_level = 0.95
        CI_label = stats.t.interval(confidence_level, len(means)-1, loc=np.mean(means), scale=stats.sem(means))
        
        # min and max fold size in terms of expected unique ID in each
        min_fold_size = np.sum([id_level_value_counter[value]/folds for value  in id_level_value_counter]) 
        max_fold_size = min_fold_size + len(id_level_value_counter)
        
        # check if expected conditions hold
        
        # checks if the sum of all unique id in all folds is like in original file
        self.assertTrue(np.sum(folds_size) == len(input_df[unique_ID].unique()))
        
        # checks if number or records in the folds in like in original file
        self.assertTrue(np.sum(records_size) == len(input_df))
        
        # checks if label balancing is distributed around the original balance
        self.assertTrue(CI_label[0] <= population_mean <= CI_label[1])
        
        # checks if in each fold number of unique ID is in the expected range
        for size in folds_size :
            self.assertTrue(min_fold_size <= size <= max_fold_size)      
        




    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()