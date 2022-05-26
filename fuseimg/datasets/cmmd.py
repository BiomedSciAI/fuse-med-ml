from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuseimg.data.ops.image_loader import OpLoadDicom
from fuseimg.data.ops.color import OpNormalizeAgainstSelfImpl
from fuseimg.data.ops.shape_ops import OpFlipBrightSideOnLeft2D , OpRemoveDarkBackgroundRectangle2D, OpResizeAndPad2D
from fuse.data import PipelineDefault, OpToTensor
from fuse.data.ops.ops_common import OpLambda
from fuseimg.data.ops.aug.color import OpAugColor
from fuseimg.data.ops.aug.geometry import OpAugAffine2D 
from fuse.data.ops.ops_aug_common import OpSample
from fuse.data.ops.ops_read import OpReadDataframe
from functools import partial
import torch
import pandas as pd
from typing import Tuple, List
import os

from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool


class CMMD:
    """
    # dataset that contains breast mamography biopsy info and metadata from chinese patients
    # Path to the stored dataset location
    # dataset should be download from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508
    # download requires NBIA data retriever https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
    # put on the folliwing in the main folder  - 
    # 1. CMMD_clinicaldata_revision.csv which is a converted version of CMMD_clinicaldata_revision.xlsx 
    # 2. folder named CMMD which is the downloaded data folder
    """
    # bump whenever the static pipeline modified
    CMMD_DATASET_VER = 0

    @staticmethod
    def static_pipeline(data_dir: str,data_source : pd.DataFrame) -> PipelineDefault:
        """
        Get suggested static pipeline (which will be cached), typically loading the data plus design choices that we won't experiment with.
        :param data_path: path to original kits21 data (can be downloaded by KITS21.download())
        """
        static_pipeline = PipelineDefault("static", [
            
            (OpReadDataframe(data_source,key_column = None , columns_to_extract = ['file','classification'] , rename_columns=dict(file="data.input.img_path",classification="data.gt.classification")), dict()), # will save image and seg path to "data.input.img_path", "data.gt.seg_path" 
            (OpLoadDicom(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
            (OpFlipBrightSideOnLeft2D(), dict(key="data.input.img")),
            (OpRemoveDarkBackgroundRectangle2D(), dict(key="data.input.img")),
            (OpNormalizeAgainstSelfImpl(), dict(key="data.input.img")),
            (OpResizeAndPad2D(), dict(key="data.input.img", resize_to=(2200, 1200), padding=(60, 60))),
            ])
        return static_pipeline

    @staticmethod
    def dynamic_pipeline():
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations. 
        """
        dynamic_pipeline = PipelineDefault("dynamic", [
            (OpToTensor(), dict(key="data.input.img",dtype=torch.float32)),
            (OpSample(OpAugAffine2D()), dict(
                            key="data.input.img",
                            rotate=Uniform(-30.0,30.0),        
                            scale=Uniform(0.9, 1.1),
                            flip=(RandBool(0.3), RandBool(0.5)),
                            translate=(RandInt(-10, 10), RandInt(-10, 10))
                        )),
            (OpSample(OpAugColor()), dict(
                        key="data.input.img",
                        gamma=Uniform(0.9, 1.1), 
                        contrast=Uniform(0.85, 1.15),
                        mul =  Uniform(0.95, 1.05),
                        add=Uniform(-0.06, 0.06)
                    )),
            (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="data.input.img")), 
            
        ])
        return dynamic_pipeline

    @staticmethod
    def create_dataset_partition(phase : str,
                                data_dir: str,
                                data_source : pd.DataFrame,
                                cache_dir : str = None,
                                restart_cache : bool = True,
                                specific_ids : List = []) :
        """
        Creates Fuse Dataset single object (either for training, validation and test or user defined set)
        :param phase:                       parition name (training / validation / test)
        :param data_dir:                    dataset root path
        :param data_source                  csv file containing all samples file paths and ground truth
        :param cache_dir:                   Optional, name of the cache folder
        :param reset_cache:                 Optional,specifies if we want to clear the cache first
        :param specific_ids                 Otional, specify which sample ids to include in this set instead of all given in dataframe
        :return: DatasetDefault object
        """


        static_pipeline = CMMD.static_pipeline(data_dir,data_source)
        dynamic_pipeline = CMMD.dynamic_pipeline()

        phase_cahce_dir = os.path.join(cache_dir,phase)                                   
        cacher = SamplesCacher(f'cmmd_cache_ver', 
            static_pipeline,
            cache_dirs=[phase_cahce_dir], restart_cache=restart_cache)   
        
        sample_ids=[id for id in data_source.index]
        if specific_ids != []:
            sample_ids = specific_ids
        my_dataset = DatasetDefault(sample_ids=sample_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
            cacher=cacher,            
        )

        my_dataset.create()
        return my_dataset
    
    