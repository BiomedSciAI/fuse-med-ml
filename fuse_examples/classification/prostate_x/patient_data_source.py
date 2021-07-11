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

from typing import List, Tuple

from fuse.data.data_source.data_source_base import FuseDataSourceBase
from fuse_examples.classification.prostate_x.data_utils import FuseProstateXUtilsData

class FuseProstateXDataSourcePatient(FuseDataSourceBase):
    def __init__(self,
                 db_path: str,
                 set_type: str,
                 db_name: str,
                 db_ver:int = 11,
                 fold_no: int=0,
                 include_gt: bool = True,
                 include_pred: bool = True,

                 ):
        """
        Fuse DataSource for ProstateX data.
        Generate sample decription per patient

        :param  set_type: 'train' 'validation' 'test'
        :param db_ver: database version
        :type include_gt: create two descriptors per patient - with 'gt' key and without
        :return list of sample descriptors

        """
        self.db_path = db_path
        self.set_type = set_type
        self.db_ver = db_ver
        self.include_gt = include_gt
        self.include_pred = include_pred
        self.db_name = db_name
        self.fold_no = fold_no
        self.desc_list = self.generate_patient_list()




    def get_samples_description(self):
        return list(self.desc_list)

    def summary(self) -> str:
        """
        See base class
        """
        summary_str = ''
        summary_str += f'Class = {type(self)}\n'
        summary_str += f'Input source = {self.set_type}\n'
        summary_str += f'Number of Patients = {len(self.desc_list)}\n'
        return summary_str


    def generate_patient_list(self) -> List[Tuple]:
        '''
        Go Over all patients and create a tuple list of (db_ver, set_type, patient_id [,'gt'])
        :return: list of patient descriptors
        '''
        data = FuseProstateXUtilsData.get_dataset(self.db_path,self.set_type, self.db_ver,self.db_name,self.fold_no)
        if self.db_name=='prostate_x':
            data_lesions = FuseProstateXUtilsData.get_lesions_prostate_x(data)

        patients = list(data_lesions['Patient ID'].unique())
        desc_list = []
        if self.include_pred:
            desc_list += [(self.db_ver, self.set_type, patient_id, 'pred') for patient_id in patients]
        if self.include_gt:
            desc_list += [(self.db_ver, self.set_type, patient_id, 'gt') for patient_id in patients]
        return desc_list

if __name__ == "__main__":
    path_to_db = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/virtual_biopsy/prostate/experiments/V1/'
    train_data_source = FuseProstateXDataSourcePatient(path_to_db,'train',db_name='tcia', db_ver='18042021',fold_no=0, include_gt=False)
