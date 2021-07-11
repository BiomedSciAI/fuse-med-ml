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

from typing import Tuple
import os
import SimpleITK as sitk
import numpy as np
import torch
import logging

from fuse.data.processor.processor_base import FuseProcessorBase

from fuse_examples.classification.prostate_x.data_utils import FuseProstateXUtilsData
from fuse_examples.classification.prostate_x.preprocessing import PROSTATEX3DVolume



class FuseProstateXPatchProcessor(FuseProcessorBase):
    def __init__(self,
                 path_to_db: str,
                 data_path: str,
                 ktrans_data_path: str,
                 db_name: str,
                 fold_no : int,
                 lsn_shape: Tuple[int, int, int] = (16, 120, 120),
                 reference_inx: int = 0,
                 ):

        # store input parameters
        self.path_to_db = path_to_db
        self.data_path = data_path
        self.ktrans_data_path = ktrans_data_path
        self.lsn_shape = lsn_shape
        self.db_name = db_name
        self.fold_no=fold_no
        self.reference_inx = reference_inx


    def __call__(self,
                 sample_desc,
                 *args, **kwargs):
        """
        Return list of samples (lesions) giving a patient level descriptor
        :param sample_desc: (db_ver, set_type, patient_id)
        :return: list of lesions, see TorchClassificationAlgo.create_lesion_sample()
        """
        samples = []

        # decode descriptor
        db_ver, set_type, patient_id,pred_or_gt = sample_desc

        # get db - lesions
        db_full = FuseProstateXUtilsData.get_dataset(self.path_to_db,set_type,db_ver,self.db_name,self.fold_no)
        db = FuseProstateXUtilsData.get_lesions_prostate_x(db_full)

        # get patient
        patient = db[db['Patient ID'] == patient_id]

        # Print patient ID
        lgr = logging.getLogger('Fuse')
        lgr.info(f'patient={patient_id}', {'color': 'magenta'})

        # ---------------Read all images and masks, and then combine them---------------

        # Read all the modalities for a given ProstateX patient ID
        # There might be multiple directories (or not)
        self.prostate_data_path = self.data_path+'/PROSTATEx/'
        patient_directories = os.listdir(os.path.join(self.prostate_data_path, patient_id))
        if len(patient_directories) != 1:
            lgr.info(' - Warning: Multiple directories!')
        patient_directories = patient_directories[0]
        images_path = os.path.join(self.prostate_data_path, patient_id, patient_directories)

        # ------------------
        # initiate volume class
        vol3D = PROSTATEX3DVolume(patient_id=patient_id, imgs_path=images_path, ktrans_data_path=self.ktrans_data_path)
        # ------------------
        # list of volumes per sequence
        vol_list = vol3D.read_prostatex_sequences_patient()
        vol_ref = vol_list[self.reference_inx]
        #------------------
        # vol_4D is multichannel volume
        vol_4D = vol3D.preprocess_sequences(vol_list, reference_inx=self.reference_inx)


        #------------------
        # each row contains one lesion
        for index, row in patient.iterrows():

            pos_orig = np.array(np.fromstring(row.values[1], dtype=np.float32, sep=' '))
            pos_vol = np.array(vol_ref.TransformPhysicalPointToContinuousIndex(pos_orig.astype(np.float64)))

            vol_cropped, mask_cropped = vol3D.crop_lesion_vol(
                vol_4D, pos_vol,vol_ref ,center_slice=pos_vol[2],
                size=(self.lsn_shape[2], self.lsn_shape[1], self.lsn_shape[0]),
                spacing=(0.5, 0.5, 3))

            vol_cropped_tmp = sitk.GetArrayFromImage(vol_cropped)
            if len(vol_cropped_tmp.shape)<4:
                vol_cropped_tmp = vol_cropped_tmp[:,:,:,np.newaxis]
                vol = np.moveaxis(vol_cropped_tmp, 3, 0)
            else:
                vol = np.moveaxis(sitk.GetArrayFromImage(vol_cropped), 3, 0)

            if sum(sum(sum(sum(np.isnan(vol))))) > 0:
                input[np.isnan(input)] = 0

            mask = sitk.GetArrayFromImage(mask_cropped)
            vol_tensor = torch.from_numpy(vol).type(torch.FloatTensor)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).type(torch.FloatTensor)

            sample = {
                'patient_num': patient_id,
                'lesion_num': row['fid'],
                'input': vol_tensor,
                'input_lesion_mask': mask_tensor,
                'ggg': row['ggg'],
                'zone': row['zone'],
                'ClinSig': row['ClinSig'],

            }

            samples.append(sample)



        return samples


if __name__ == "__main__":
    path_to_db = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/virtual_biopsy/prostate/experiments/V1/'
    dataset = 'prostate_x'
    if dataset=='prostate_x':
    # for ProstateX
        path_to_dataset = '/projects/msieve/MedicalSieve/PatientData/ProstateX/manifest-A3Y4AE4o5818678569166032044/'
        prostate_data_path = path_to_dataset
        Ktrain_data_path = path_to_dataset + '/ProstateXKtrains-train-fixed/'
        sample = ('18052021', 'train', 'ProstateX-0148', 'pred')

        a = FuseProstateXPatchProcessor(path_to_db,data_path=prostate_data_path,ktrans_data_path=Ktrain_data_path,db_name=dataset,fold_no=1,lsn_shape=(13, 74, 74),reference_inx=0)
        samples = a.__call__(sample)
        a = 1
