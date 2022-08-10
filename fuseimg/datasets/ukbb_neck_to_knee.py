from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuseimg.data.ops.color import OpNormalizeAgainstSelf
from fuse.data import PipelineDefault, OpToTensor
from fuse.data.ops.ops_common import OpLambda
from fuseimg.data.ops.aug.color import OpAugColor
from fuseimg.data.ops.aug.geometry import OpAugAffine2D 
from fuse.data.ops.ops_aug_common import OpSample, OpRandApply
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_cast import OpToNumpy
from fuse.data.ops.op_base import OpBase
from fuse.utils import NDict
from functools import partial
from typing import Hashable, List, Optional, Sequence
import torch
import pandas as pd
import numpy as np
import pydicom
import io
import os
import multiprocessing
from pathlib import Path
from fuse.data.utils.sample import get_sample_id
import zipfile
# from fuseimg.data.ops import ops_mri
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool
import SimpleITK as sitk
import tempfile
import shutil
import skimage
import skimage.transform
from glob import glob

from matplotlib import pyplot as plt
def dump(img, filename, slice):
    plt.imshow(img[slice,:,:], interpolation='nearest')
    plt.savefig(filename)
    return img

class OpUKBBSampleIDDecode(OpBase):
    '''
    decodes sample id into image and segmentation filename
    '''

    def __call__(self, sample_dict: NDict) -> NDict:
        '''
        
        '''
        sid = get_sample_id(sample_dict)
        
        img_filename_key = 'data.input.img_path'
        sample_dict[img_filename_key] =   sid

        return sample_dict

class OpLoadUKBBZip(OpBase):
    '''
    loads a zip and select a sequence and a station from it
    '''
    def __init__(self, dir_path: str, **kwargs):
        super().__init__(**kwargs)
        self._dir_path = dir_path

    def __call__(self, sample_dict: NDict, series_config : NDict, key_in:str, key_out: str, unique_id_out: str) -> NDict:
        '''
        
        '''
        scans = []
        zip_filenames = glob(os.path.join(self._dir_path,sample_dict[key_in]))
        if len(zip_filenames) >1:
            raise NotImplementedError(f"{sample_dict[key_in]} has more then one match. Currently not supported")
        zip_filename = zip_filenames[0]
        try:
            zip_file = zipfile.ZipFile(zip_filename)
        except:
            print("error in opening",zip_filename, os.path.exists(zip_filename))
            return None
        filenames_list = [f.filename for f in zip_file.infolist() if '.dcm' in f.filename]
        
        for dicom_file in filenames_list:
            with zip_file.open(dicom_file) as f:
                dcm = pydicom.read_file(io.BytesIO(f.read()))
                scans.append({'file': zip_filename.split("/")[-1], 'dcm_unique': dcm[0x0020000e].value, 'time':dcm[0x00080031].value, 'series': dcm[0x0008103e].value})
                
        dicom_tags = pd.DataFrame(scans)
        dicom_tags['n_slices'] = dicom_tags.groupby(dicom_tags.columns.to_list())['file'].transform('size')
        dicom_tags = dicom_tags.drop_duplicates()
        dicom_tags = dicom_tags.sort_values(by=['time'])
        if series_config['series'] in ['Dixon_noBH_in', 'Dixon_noBH_opp', 'Dixon_noBH_F', 'Dixon_noBH_W', 'Dixon_BH_17s_in', 'Dixon_BH_17s_opp', 'Dixon_BH_17s_F', 'Dixon_BH_17s_W'] :
            if len(dicom_tags) != 24:
                print(zip_filename, "has missing/extra sequences ",len(dicom_tags),"instead of 24")
                return None
            station_list = []
            for i in range(6) :
                for j in range(4) :
                        station_list.append(i+1)
            dicom_tags['station'] = station_list
            try :
                dcm_unique = dicom_tags[(dicom_tags['station'] == series_config['station']) & (dicom_tags['series'] == series_config['series'])]['dcm_unique'].iloc[0]
            except:
                print("requested file",zip_filename,"series description",series_config, "not found!!!")
                return None
        else:
            try:
                dcm_unique = dicom_tags[dicom_tags['series'] == series_config['series']]['dcm_unique'].iloc[0]
            except:
                print("requested file",zip_filename,"series description",series_config, "not found!!!")
                return None
        dirpath = tempfile.mkdtemp()
        # ... do stuff with dirpath
        for dicom_file in filenames_list:
            with zip_file.open(dicom_file) as f:
                if pydicom.read_file(io.BytesIO(f.read()))[0x0020000e].value  == dcm_unique :
                    zip_file.extract(dicom_file, path=dirpath)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dirpath)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        numpy_img = sitk.GetArrayFromImage(image)
        sample_dict[key_out] = numpy_img
        sample_dict[unique_id_out] = dcm_unique
        shutil.rmtree(dirpath)
        return sample_dict
    
class OpLoadCenterPoint(OpBase):
    '''
    loads a zip and select a sequence and a station from it
    '''
    def __init__(self, dir_path: str, **kwargs):
        super().__init__(**kwargs)
        self._dir_path = dir_path

    def __call__(self, sample_dict: NDict, key_in:str, key_out: str) -> NDict:
        '''
        
        '''
        filename = os.path.join(self._dir_path,sample_dict[key_in])
        f = open(filename, 'r').read()
        sample_dict[key_out]= np.loadtxt(f, delimiter=',', usecols=(0, 2))
        return sample_dict
class OpLoadVolumeAroundCenterPoint(OpBase):
    '''
    loads a volume (defined by radiuses) from given image around the given center point 
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample_dict: NDict , key_in:str, key_out: str, centerpoint : str = None, radiuses : np.array= np.array([5.0,30.0,30.0])) -> NDict:
        '''
        
        '''
        img = sample_dict[key_in]
        if centerpoint == None :
            center_point_cord = [i/2 for i in img.shape]
        else:
            center_point_cord = sample_dict[centerpoint]
        bouding_box_indices =  [(max(int((center_point_cord[i]-radiuses[i])),0),min(int((center_point_cord[i]+radiuses[i])),img.shape[i]-1)) for i in range(3)]
        img = img[bouding_box_indices[0][0]:bouding_box_indices[0][1],bouding_box_indices[1][0]:bouding_box_indices[1][1],bouding_box_indices[2][0]:bouding_box_indices[2][1]]
        sample_dict[key_out] = img
        return sample_dict
class UKBB:
    """
    # dataset that contains MRI nech-to-knee  and metadata from UK patients
    # Path to the stored dataset location
    # put on the folliwing in the main folder  - 
    # 1. label.csv 
    # 2. folder named body-mri-data which is the downloaded data folder
    """
    # bump whenever the static pipeline modified
    UKBB_DATASET_VER = 0

    @staticmethod
    def download(path: str) -> None:
        '''
        Automatic download is not supported, please follow instructions in STOIC21 class header to download
        '''
        assert len(UKBB.sample_ids(path)) > 0, "automatic download is not supported, please follow instructions in STOIC21 class header to download"


    @staticmethod
    def sample_ids(path: str):
        return UKBB.get_existing_sample_ids(path)

    @staticmethod
    def get_existing_sample_ids(path: str):
        """
        get all the sample ids that have a zip file in the specified path
        """
        existing_files = glob(os.path.join(path, "*_*_*_0.zip"))
        existing_sample_id_fields = [f.split("_") for f in existing_files]
        existing_sample_ids = set([a[0] + "_*_" + a[2] + "_" + a[3] for a in existing_sample_id_fields])

        return existing_sample_ids
    @staticmethod
    def static_pipeline(data_dir: str, series_config: NDict, centerpoint_dir: str = None) -> PipelineDefault:
        """
        Get suggested static pipeline (which will be cached), typically loading the data plus design choices that we won't experiment with.
        :param data_path: path to original kits21 data (can be downloaded by KITS21.download())
        """
        static_pipeline = PipelineDefault("cmmd_static", [
         # decoding sample ID
            (OpUKBBSampleIDDecode(), dict()), # will save image and seg path to "data.input.img_path", "data.gt.seg_path"    
            (OpLoadUKBBZip(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", unique_id_out="data.ID", series_config=series_config)),
            # (OpLoadCenterPoint(centerpoint_dir), dict(key_in="data.input.img_path", key_out="data.input.centerpoint")),
            # (OpLoadVolumeAroundCenterPoint(), dict(key_in="data.input.img", key_out="data.input.img" , centerpoint = "data.input.centerpoint")),
            (OpLambda(partial(skimage.transform.resize,
                                                            output_shape=(44, 174, 224),
                                                            mode='reflect',
                                                            anti_aliasing=True,
                                                            preserve_range=True)), dict(key="data.input.img")),
            (OpNormalizeAgainstSelf(), dict(key="data.input.img")),
            (OpToNumpy(), dict(key='data.input.img', dtype=np.float32)), 
            # (OpLambda(partial(dump, filename="first.png", slice = 25)), dict(key="data.input.img")),
            ])
        return static_pipeline

    @staticmethod
    def dynamic_pipeline(data_source : pd.DataFrame, target: str, train: bool = False):
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations. 
        :param train : True iff we request dataset for train purpouse
        """
        dynamic_pipeline = PipelineDefault("cmmd_dynamic", [
            (OpReadDataframe(data_source,
                    key_column="file_pattern", columns_to_extract=['file_pattern','patient_id', target],
                    rename_columns={'patient_id' :"data.patientID", target: "data.gt.classification" }), dict()),
            (OpToTensor(), dict(key="data.input.img",dtype=torch.float32)),
            (OpToTensor(), dict(key="data.gt.classification", dtype=torch.long)),
            (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="data.input.img")) ])
        # augmentation
        if train:
            dynamic_pipeline.extend([ 
                (OpLambda(partial(torch.squeeze, dim=0)), dict(key="data.input.img")),  

                # affine augmentation - will apply the same affine transformation on each slice
                (OpRandApply(OpSample(OpAugAffine2D()), 0.5), dict(
                    key="data.input.img",
                    rotate=Uniform(-180.0,180.0),        
                    scale=Uniform(0.8, 1.2),
                    flip=(RandBool(0.5), RandBool(0.5)),
                    translate=(RandInt(-15, 15), RandInt(-15, 15))
                )),
                
                # color augmentation - check if it is useful in CT images
                # (OpSample(OpAugColor()), dict(
                #     key="data.input.img",
                #     gamma=Uniform(0.8,1.2), 
                #     contrast=Uniform(0.9,1.1),
                #     add=Uniform(-0.01, 0.01)
                # )),

                # add channel dimension -> [C=1, D, H, W]
                (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="data.input.img")),
                  
        ])

        return dynamic_pipeline


    
    @staticmethod
    def dataset(
                data_dir: str,
                target: str,
                series_config: NDict,
                input_source_gt: pd.DataFrame = None,
                cache_dir : str = None,
                reset_cache : bool = True,
                num_workers:int = 10,
                sample_ids: Optional[Sequence[Hashable]] = None,
                train: bool = False) :
        """
        Creates Fuse Dataset single object (either for training, validation and test or user defined set)
        
        :param data_dir:                    dataset root path
        :param target                       target name used from the ground truth dataframe
        :param series_config                configuration of the selected series from the ukbb zip
        :param input_source_gt              dataframe with ground truth file
        :param cache_dir:                   Optional, name of the cache folder
        :param reset_cache:                 Optional,specifies if we want to clear the cache first
        :param num_workers: number of processes used for caching 
        :param sample_ids: dataset including the specified sample_ids or None for all the samples. sample_id is case_{id:05d} (for example case_00001 or case_00100).
        :param train: True if used for training  - adds augmentation operations to the pipeline
        :return: DatasetDefault object
        """


        if sample_ids is None:
            sample_ids = UKBB.sample_ids(data_dir)


        static_pipeline = UKBB.static_pipeline(data_dir, series_config)
        dynamic_pipeline = UKBB.dynamic_pipeline(input_source_gt, target,train=train)
                                
        cacher = SamplesCacher(f'cmmd_cache_ver', 
            static_pipeline,
            cache_dirs=[cache_dir],
            restart_cache=reset_cache,
            audit_first_sample=False, audit_rate=None,
            workers=num_workers)   
        
        my_dataset = DatasetDefault(sample_ids=sample_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
            cacher=cacher,            
        )

        my_dataset.create(num_workers = num_workers)
        return my_dataset
