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

import pickle
import pandas as pd
import os

class FuseProstateXUtilsData:
    @staticmethod
    def get_dataset(path_to_db: str,set_type: str, db_ver: int,db_name: str,fold_no: int):
        db_name = os.path.join(path_to_db,f'dataset_{db_name}_folds_ver{db_ver}_seed1.pickle')
        with open(db_name, 'rb') as infile:
            db = pickle.load(infile)


        if set_type == 'train':
            other_folds = list(set(range(0,len(db)))-set([fold_no]))
            for i,f in enumerate(other_folds):
                if i==0:
                    data = db['data_fold' + str(f)]
                else:
                    data = pd.concat([data,db['data_fold' + str(f)]],join='inner')

        elif set_type == 'validation':
            data = db['data_fold'+str(fold_no)]
        elif set_type == 'test':
            data = db['test_data']
        else:
            raise Exception(f'Unexpected set type {set_type}')
        return data

    def get_lesions_prostate_x(data: pd.DataFrame):
        outlier_list = []
        lesion_data = data[~data['Patient ID'].isin(outlier_list)]
        return lesion_data


if __name__ == "__main__":
    path_to_db = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/virtual_biopsy/prostate/experiments/V1/'
    # data = FuseCAPVUtilsData.get_dataset(path_to_db=path_to_db,set_type='train', db_ver=18042021,db_name='tcia',fold_no=0)
    # data_lesion = FuseCAPVUtilsData.get_lesions(data)

    data = FuseProstateXUtilsData.get_dataset(path_to_db=path_to_db, set_type='train', db_ver=29042021, db_name='prostate_x',fold_no=0)
    data_lesion = FuseProstateXUtilsData.get_lesions_prostate_x(data)