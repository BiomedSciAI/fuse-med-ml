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
from scipy.ndimage.morphology import binary_dilation

from fuse.data.processor.processor_base import FuseProcessorBase

from fuse_examples.classification.prostate_x.data_utils import FuseProstateXUtilsData
# from fuse_examples.classification.prostate_x.processor_dicom_mri import FuseDicomMRIProcessor
from fuse.data.processor.processor_dicom_mri import FuseDicomMRIProcessor


class FuseProstateXPatchProcessor(FuseProcessorBase):
    """
    This processor crops the lesion volume from within 4D MRI volume base on
    lesion location as appears in the database.
    :returns a sample that includes:
                'patient_num': patient id
                'lesion_num': one MRI volume may include more than one lesion
                'input': vol_tensor as extracted from MRI volume processor
                'input_lesion_mask': mask_tensor,
                'ggg': row['ggg']: in prostate - lesion grade
                'zone': row['zone']: zone in prostate
                'ClinSig': row['ClinSig']: Clinical significant ( 0 for benign and 3+3 lesions, 1 for rest)
    """
    def __init__(self,
                 vol_processor: FuseDicomMRIProcessor = FuseDicomMRIProcessor(),
                 path_to_db: str = None,
                 data_path: str = None,
                 ktrans_data_path: str = None,
                 db_name: str = None,
                 db_version: str = None,
                 fold_no : int = None,
                 lsn_shape: Tuple[int, int, int] = (16, 120, 120),
                 lsn_spacing: Tuple[float, float, float] = (3, 0.5, 0.5),
                 ):
        """
        :param vol_processor - extracts 4D tensor from path to MRI dicoms
        :param path_to_db: path to data pickle
        :param data_path: path to directory in which dicom data is located
        :param ktrans_data_path: path to directory of Ktrans seq (prostate x)
        :param db_name: 'prostatex' for this example
        :param fold_no: cross validation fold
        :param lsn_shape: shape of volume to extract from full volume (pixels)
        :param lsn_spacing: spacing of volume to extract from full volume (mm)
        """

        # store input parameters
        self.vol_processor = vol_processor
        self.path_to_db = path_to_db
        self.data_path = data_path
        self.ktrans_data_path = ktrans_data_path
        self.lsn_shape = lsn_shape
        self.lsn_spacing = lsn_spacing
        self.db_name = db_name
        self.db_ver = db_version
        self.fold_no=fold_no
        self.prostate_data_path = os.path.join(self.data_path,'PROSTATEx/')



    # ========================================================================
    def create_resample(self,vol_ref:sitk.sitkFloat32, interpolation: str, size:Tuple[int,int,int], spacing: Tuple[float,float,float]):
        """
        create_resample create resample operator
        :param vol_ref: sitk vol to use as a ref
        :param interpolation:['linear','nn','bspline']
        :param size: in pixels ()
        :param spacing: in mm ()
        :return: resample sitk operator
        """

        if interpolation == 'linear':
            interpolator = sitk.sitkLinear
        elif interpolation == 'nn':
            interpolator = sitk.sitkNearestNeighbor
        elif interpolation == 'bspline':
            interpolator = sitk.sitkBSpline

        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(vol_ref)
        resample.SetOutputSpacing(spacing)
        resample.SetInterpolator(interpolator)
        resample.SetSize(size)
        return resample

    # ========================================================================
    def apply_resampling(self,img:sitk.sitkFloat32, mask:sitk.sitkFloat32,
                             spacing: Tuple[float,float,float] =(0.5, 0.5, 3), size: Tuple[int,int,int] =(160, 160, 32),
                             transform:sitk=None, interpolation:str='bspline',
                             label_interpolator:sitk=sitk.sitkLabelGaussian,
                             ):

        ref = img if img != [] else mask
        size = [int(s) for s in size]
        resample = self.create_resample(ref, interpolation, size=size, spacing=spacing)

        if ~(transform is None):
            resample.SetTransform(transform)
        img_r = resample.Execute(img)

        resample.SetInterpolator(label_interpolator)
        mask_r = resample.Execute(mask)


        return img_r, mask_r

    # ========================================================================
    def crop_lesion_vol(self,vol:sitk.sitkFloat32, position:Tuple[float,float,float], ref:sitk.sitkFloat32, size:Tuple[int,int,int]=(160, 160, 32),
                        spacing:Tuple[int,int,int]=(1, 1, 3), center_slice=None):
        """
        crop_lesion_vol crop tensor around position
        :param vol: vol to crop
        :param position: point to crop around
        :param ref: reference volume
        :param size: size in pixels to crop
        :param spacing: spacing to resample the col
        :param center_slice: z coordinates of position
        :return: cropped volume
        """

        def get_lesion_mask(position, ref):
            mask = np.zeros_like(sitk.GetArrayViewFromImage(ref), dtype=np.uint8)

            coords = np.round(position[::-1]).astype(np.int)
            mask[coords[0], coords[1], coords[2]] = 1
            mask = binary_dilation(mask, np.ones((3, 5, 5))) + 0
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.CopyInformation(ref)

            return mask_sitk

        mask = get_lesion_mask(position, ref)

        vol.SetOrigin((0,) * 3)
        mask.SetOrigin((0,) * 3)
        vol.SetDirection(np.eye(3).flatten())
        mask.SetDirection(np.eye(3).flatten())

        ma_centroid = mask > 0.5
        label_analysis_filer = sitk.LabelShapeStatisticsImageFilter()
        label_analysis_filer.Execute(ma_centroid)
        centroid = label_analysis_filer.GetCentroid(1)
        offset_correction = np.array(size) * np.array(spacing)/2
        corrected_centroid = np.array(centroid)
        corrected_centroid[2] = center_slice * np.array(spacing[2])
        offset = corrected_centroid - np.array(offset_correction)

        translation = sitk.TranslationTransform(3, offset)
        img, mask = self.apply_resampling(vol, mask, spacing=spacing, size=size, transform=translation)

        return img, mask



    # ========================================================================
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
        patient_id= sample_desc

        # ========================================================================
        # get db - lesions
        db_full = FuseProstateXUtilsData.get_dataset(self.path_to_db,'other',self.db_ver,self.db_name,self.fold_no)
        db = FuseProstateXUtilsData.get_lesions_prostate_x(db_full)

        # ========================================================================
        # get patient
        patient = db[db['Patient ID'] == patient_id]
        # ========================================================================
        lgr = logging.getLogger('Fuse')
        lgr.info(f'patient={patient_id}', {'color': 'magenta'})


        # ========================================================================
        # all seq paths for a certain patient


        patient_directories = os.listdir(os.path.join(self.prostate_data_path, patient_id))
        patient_directories = patient_directories[0]
        images_path = os.path.join(self.prostate_data_path, patient_id, patient_directories)

        # ========================================================================
        # vol_4D is multichannel volume (z,x,y,chan(sequence))
        vol_4D,vol_ref = self.vol_processor((images_path,self.ktrans_data_path,patient_id))

        # ========================================================================
        # each row contains one lesion, iterate over lesions

        for index, row in patient.iterrows():
            #read original position
            pos_orig = np.array(np.fromstring(row.values[1], dtype=np.float32, sep=' '))
            # transform to pixel coordinate in ref coords
            pos_vol = np.array(vol_ref.TransformPhysicalPointToContinuousIndex(pos_orig.astype(np.float64)))
            # crop lesion vol
            vol_cropped, mask_cropped = self.crop_lesion_vol(
                vol_4D, pos_vol,vol_ref ,center_slice=pos_vol[2],
                size=(self.lsn_shape[2], self.lsn_shape[1], self.lsn_shape[0]),
                spacing=(self.lsn_spacing[2], self.lsn_spacing[1], self.lsn_spacing[0]))

            vol_cropped_tmp = sitk.GetArrayFromImage(vol_cropped)
            if len(vol_cropped_tmp.shape)<4:
                # fix dimensions in case of one seq
                vol_cropped_tmp = vol_cropped_tmp[:,:,:,np.newaxis]
                vol = np.moveaxis(vol_cropped_tmp, 3, 0)
            else:
                vol = np.moveaxis(sitk.GetArrayFromImage(vol_cropped), 3, 0)

            if np.isnan(vol).any():
                input[np.isnan(input)] = 0

            mask = sitk.GetArrayFromImage(mask_cropped)
            vol_tensor = torch.from_numpy(vol).type(torch.FloatTensor)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).type(torch.FloatTensor)

            # sample
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
    import matplotlib.pyplot as plt
    import pandas as pd

    path_to_db = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/virtual_biopsy/prostate/experiments/V4/'
    dataset = 'prostate_x'
    if dataset=='prostate_x':
    # for ProstateX
        path_to_dataset = '/projects/msieve/MedicalSieve/PatientData/ProstateX/manifest-A3Y4AE4o5818678569166032044/'
        prostate_data_path = path_to_dataset
        Ktrain_data_path = path_to_dataset + '/ProstateXKtrains-train-fixed/'
        sample = ('29062021', 'train', 'ProstateX-0148', 'pred')

        a = FuseProstateXPatchProcessor(vol_processor=FuseDicomMRIProcessor(reference_inx=0),path_to_db = path_to_db,
                                        data_path=prostate_data_path,ktrans_data_path=Ktrain_data_path,
                                        db_name=dataset,fold_no=1,lsn_shape=(13, 74, 74))
        samples = a.__call__(sample)
        l_seq = pd.read_csv('/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/virtual_biopsy/prostate/prostate_x/metadata.csv')
        for sample_id in list(l_seq['Subject ID'].unique()):
            # sample_id = 'ACRIN-6698-760011'
            sample = ('29062021', 'validation', sample_id, 'pred')
            samples = a.__call__(sample)
            if len(samples)==0:
                sample = ('29062021', 'train', sample_id, 'pred')
                samples = a.__call__(sample)

            path2save = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/virtual_biopsy/prostate/prostate_x/data_visualization/'
            fix, ax = plt.subplots(nrows=5, ncols=13, sharex=True, sharey=True)
            for idx in range(5):
                for jdx in range(13):
                    ll = samples[0]['input'].cpu().detach().numpy()[idx, jdx, :, :]
                    ax[idx, jdx].imshow(ll, cmap='gray')
            fix.suptitle(sample_id)
            fix.savefig(path2save + sample_id + '.jpg')