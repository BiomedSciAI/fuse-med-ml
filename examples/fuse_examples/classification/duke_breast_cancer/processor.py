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
import cv2
from scipy.ndimage.morphology import binary_dilation
from fuse.data.processor.processor_base import ProcessorBase
from fuse.data.processor.processor_dicom_mri import DicomMRIProcessor

from fuse_examples.classification.prostate_x.data_utils import ProstateXUtilsData


class PatchProcessor(ProcessorBase):
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
                 vol_processor: DicomMRIProcessor = DicomMRIProcessor(),
                 path_to_db: str = None,
                 data_path: str = None,
                 ktrans_data_path: str = None,
                 db_name: str = None,
                 db_version: str = None,
                 fold_no : int = None,
                 lsn_shape: Tuple[int, int, int] = (16, 120, 120),
                 lsn_spacing: Tuple[float, float, float] = (3, 0.5, 0.5),
                 longtd_inx: int = 0,
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
        self.prostate_data_path = self.data_path
        self.longtd_inx = longtd_inx


    # ========================================================================
    def create_resample(self,vol_ref:sitk.sitkFloat32, interpolation: str, size:tuple, spacing: tuple):
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
    def crop_lesion_vol_mask_based(self,vol:sitk.sitkFloat32, position:tuple, ref:sitk.sitkFloat32, size:Tuple[int,int,int]=(160, 160, 32),
                        spacing:Tuple[int,int,int]=(1, 1, 3), mask_inx = -1,is_use_mask=True):
        """
        crop_lesion_vol crop tensor around position
        :param vol: vol to crop
        :param position: point to crop around
        :param ref: reference volume
        :param size: size in pixels to crop
        :param spacing: spacing to resample the col
        :param center_slice: z coordinates of position
        :param mask_inx: channel index in which mask is located default: last channel
        :param is_use_mask: use mask to define crop bounding box
        :return: cropped volume
        """

        margin = [20,20,0]
        vol_np = sitk.GetArrayFromImage(vol)
        if is_use_mask:

            mask = sitk.GetArrayFromImage(vol)[:,:,:,mask_inx]
            mask_bool = np.zeros(mask.shape).astype(int)
            mask_bool[mask>0.01]=1
            mask_final = sitk.GetImageFromArray(mask_bool)
            mask_final.CopyInformation(ref)

            lsif = sitk.LabelShapeStatisticsImageFilter()
            lsif.Execute(mask_final)
            bounding_box = np.array(lsif.GetBoundingBox(1))
            vol_np[:, :, :, mask_inx] = mask_bool
        else:
            bounding_box = np.array([int(position[0]) - int(size[0] / 2),
                           int(position[1]) - int(size[1] / 2),
                           int(position[2]) - int(size[2] / 2),
                           size[0],
                           size[1],
                           size[2]
                           ])
        # in z use a fixed number of slices,based on position
        bounding_box[-1] = size[2]
        bounding_box[2] = int(position[2]) - int(size[2]/2)

        bounding_box_size = bounding_box[3:5][np.argmax(bounding_box[3:5])]
        dshift = bounding_box[3:5] - bounding_box_size
        dshift = np.append(dshift,0)

        ijk_min_bound = np.maximum(bounding_box[0:3]+dshift - margin,0)
        ijk_max_bound = np.maximum(bounding_box[0:3]+dshift+[bounding_box_size,bounding_box_size,bounding_box[-1]] + margin,0)



        vol_np_cropped = vol_np[ijk_min_bound[2]:ijk_max_bound[2],ijk_min_bound[1]:ijk_max_bound[1],ijk_min_bound[0]:ijk_max_bound[0],:]
        vol_np_resized = np.zeros((size[2],size[0],size[1],vol_np_cropped.shape[-1]))
        for si in range(vol_np_cropped.shape[0]):
            for ci in range(vol_np_cropped.shape[-1]):
                vol_np_resized[si,:,:,ci] = cv2.resize(vol_np_cropped[si, :,:, ci], (size[0],size[1]), interpolation=cv2.INTER_AREA)

        img = sitk.GetImageFromArray(vol_np_resized)
        mask = sitk.GetImageFromArray(vol_np_resized[:,:,:,mask_inx])

        return img, mask


    def get_zeros_vol(self,vol):
        if vol.GetNumberOfComponentsPerPixel()>1:
            ref_zeros_vol = sitk.VectorIndexSelectionCast(vol,0)
        else:
            ref_zeros_vol = vol
        zeros_vol = np.zeros_like(sitk.GetArrayFromImage(ref_zeros_vol))
        zeros_vol = sitk.GetImageFromArray(zeros_vol)
        zeros_vol.CopyInformation(ref_zeros_vol)
        return zeros_vol

    def extract_mask_from_annotation(self,vol_ref,bbox_coords):
        xstart = bbox_coords[0]
        ystart = bbox_coords[1]
        zstart = bbox_coords[2]
        xsize = bbox_coords[3]
        ysize = bbox_coords[4]
        zsize = bbox_coords[5]

        mask = self.get_zeros_vol(vol_ref)
        mask_np = sitk.GetArrayFromImage(mask)
        mask_np[zstart:zstart+zsize,ystart:ystart+ysize,xstart:xstart+xsize] = 1.0
        return mask_np

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
        patient_id = sample_desc

        # ========================================================================
        # get db - lesions
        db_full = ProstateXUtilsData.get_dataset(self.path_to_db,'other',self.db_ver,self.db_name,self.fold_no)
        db = ProstateXUtilsData.get_lesions_prostate_x(db_full)

        # ========================================================================
        # get patient
        patient = db[db['Patient ID'] == patient_id]
        # ========================================================================
        lgr = logging.getLogger('Fuse')
        lgr.info(f'patient={patient_id}', {'color': 'magenta'})


        # ========================================================================
        # all seq paths for a certain patient


        patient_directories = os.path.join(os.path.join(self.prostate_data_path, patient_id),patient['ser_name_T'+str(self.longtd_inx)].values[0][2:-2])
        images_path = os.path.join(self.prostate_data_path, patient_id, patient_directories)

        # ========================================================================
        # vol_4D is multichannel volume (z,x,y,chan(sequence))
        vol_4D,vol_ref = self.vol_processor((images_path,self.ktrans_data_path,patient_id))

        # ========================================================================
        # each row contains one lesion, iterate over lesions

        for index, row in patient.iterrows():
            #read original position
            pos_orig = np.fromstring(row['centroid_T'+str(self.longtd_inx)][1:-1], dtype=np.float32, sep=',')

            # transform to pixel coordinate in ref coords
            pos_vol = np.array(vol_ref.TransformPhysicalPointToContinuousIndex(pos_orig.astype(np.float64)))

            vol_4d_tmp = sitk.GetArrayFromImage(vol_4D)
            if sum(sum(sum(vol_4d_tmp[:,:,:,-1])))==0:
                bbox_coords = np.fromstring(row['bbox_T' +str(self.longtd_inx)][1:-1],dtype = np.int32,sep=',')
                mask = self.extract_mask_from_annotation(vol_ref,bbox_coords)
                vol_4d_tmp[:,:,:,-1] = mask
                vol_4d_new = sitk.GetImageFromArray(vol_4d_tmp)
                vol_4D = vol_4d_new



                # crop lesion vol - resized to lsn_shape
            vol_cropped_orig, mask_cropped_orig = self.crop_lesion_vol_mask_based(
                    vol_4D, pos_vol, vol_ref,
                    size=(2*self.lsn_shape[2], 2*self.lsn_shape[1], self.lsn_shape[0]),
                    spacing=(self.lsn_spacing[2], self.lsn_spacing[1], self.lsn_spacing[0]), mask_inx=-1,is_use_mask=False)

            # crop lesion vol
            vol_cropped, mask_cropped = self.crop_lesion_vol_mask_based(
                vol_4D, pos_vol,vol_ref ,
                size=(self.lsn_shape[2], self.lsn_shape[1], self.lsn_shape[0]),
                spacing=(self.lsn_spacing[2], self.lsn_spacing[1], self.lsn_spacing[0]), mask_inx = -1,is_use_mask=True)


            vol_cropped_tmp = sitk.GetArrayFromImage(vol_cropped)
            vol_cropped_orig_tmp = sitk.GetArrayFromImage(vol_cropped_orig)
            if len(vol_cropped_tmp.shape)<4:
                # fix dimensions in case of one seq
                vol_cropped_tmp = vol_cropped_tmp[:,:,:,np.newaxis]
                vol = np.moveaxis(vol_cropped_tmp, 3, 0)
                vol_cropped_orig_tmp = vol_cropped_orig_tmp[:, :, :, np.newaxis]
                vol_orig = np.moveaxis(vol_cropped_orig_tmp, 3, 0)
            else:
                vol = np.moveaxis(sitk.GetArrayFromImage(vol_cropped), 3, 0)
                vol_orig = np.moveaxis(sitk.GetArrayFromImage(vol_cropped_orig), 3, 0)

            if np.isnan(vol).any():
                input[np.isnan(input)] = 0

            mask = sitk.GetArrayFromImage(mask_cropped)
            vol_tensor = torch.from_numpy(vol).type(torch.FloatTensor)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).type(torch.FloatTensor)

            mask_orig = sitk.GetArrayFromImage(mask_cropped_orig)
            vol_tensor_orig = torch.from_numpy(vol_orig).type(torch.FloatTensor)
            mask_tensor_orig = torch.from_numpy(mask_orig).unsqueeze(0).type(torch.FloatTensor)

            # sample
            sample = {
                'patient_num': patient_id,
                'input': vol_tensor,
                'input_orig': vol_tensor_orig,
                'input_lesion_mask': mask_tensor,
                'add_data':row,

            }

            samples.append(sample)



        return samples


if __name__ == "__main__":
    from fuse_examples.classification.duke_breast_cancer.dataset import process_mri_series

    path_to_db = '.'
    root_data = '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'

    seq_dict,SER_INX_TO_USE,exp_patients,_,_ = process_mri_series(root_data+'/metadata.csv')
    mri_vol_processor = DicomMRIProcessor(seq_dict=seq_dict,
                                              seq_to_use=['DCE_mix_ph1',
                                                          'DCE_mix_ph2',
                                                          'DCE_mix_ph3',
                                                          'DCE_mix_ph4',
                                                          'DCE_mix',
                                                          'DCE_mix_ph',
                                                          'MASK'],
                                              subseq_to_use=['DCE_mix_ph2', 'MASK'],
                                              ser_inx_to_use=SER_INX_TO_USE,
                                              exp_patients=exp_patients,
                                              reference_inx=0,
                                              use_order_indicator=False)

    a = PatchProcessor(vol_processor=mri_vol_processor,
                           path_to_db=path_to_db,
                           data_path=root_data + 'Duke-Breast-Cancer-MRI',
                           ktrans_data_path='',
                           db_name='DUKE',db_version='11102021TumorSize',
                           fold_no=0, lsn_shape=(9, 100, 100), lsn_spacing=(1, 0.5, 0.5))



    sample = 'Breast_MRI_900'
    samples = a.__call__(sample)