import os
from sqlite3 import SQLITE_CREATE_TEMP_TABLE
import requests
from zipfile import ZipFile
import io
import wget
import logging
from typing import Hashable, Optional, Sequence, List


from fuse.data import DatasetDefault
from fuse.data.utils.sample import get_sample_id
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.op_base import OpBase
from fuse.data.datasets.caching.samples_cacher import SamplesCacher

from fuse.utils import NDict

from fuseimg.data.ops.image_loader import OpLoadRGBImage
from fuseimg.data.ops.color import OpNormalizeAgainstSelfImpl
from fuseimg.data.ops.aug.geometry import OpResize2D

class OpSkinLesionSampleIDDecode(OpBase):

    def __call__(self, sample_dict: NDict, op_id: Optional[str]) -> NDict:
        """
        
        """
        sid = get_sample_id(sample_dict)

        img_filename_key = 'data.input.img_path'
        sample_dict[img_filename_key] =   sid # the sample id is the image's filename itself

        return sample_dict


class SkinLesion:
    """
    TODO: describe
    """
    # bump whenever the static pipeline modified
    DATASET_VER = 0

    @staticmethod
    def download(data_path: str) -> None:
        """
        Download images and metadata from ISIC challenge
        :param data_path: path where data should be located
        """
        lgr = logging.getLogger('Fuse')
    
        path = os.path.join(data_path, 'ISIC2019/ISIC_2019_Training_Input')
        print(f"Training Input Path: {os.path.abspath(path)}")
        if not os.path.exists(path):
            lgr.info('\nExtract ISIC-2019 training input ... (this may take a few minutes)')

            url = 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip'
            wget.download(url, ".")
            
            with ZipFile("ISIC_2019_Training_Input.zip", 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(path=os.path.join(data_path, 'ISIC2019'))

            lgr.info('Extracting ISIC-2019 training input: done')

        path = os.path.join(data_path, 'ISIC2019/ISIC_2019_Training_GroundTruth.csv')

        if not os.path.exists(path):
            lgr.info('\nExtract ISIC-2019 training gt ... (this may take a few minutes)')

            url = 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv'
            wget.download(url, path)

            lgr.info('Extracting ISIC-2019 training gt: done')

    @staticmethod
    def sample_ids(data_dir: str) -> List[str]:
        """
        get all the sample ids in trainset
        sample_id is case_{id:05d} (for example case_00001 or case_00100)
        """
        test_mode = True
        samples = [f for f in os.listdir(data_dir) if f.split(".")[-1] == 'jpg']
        if test_mode:
            return ['ISIC_0000000.jpg']
        return samples

    
    @staticmethod
    def static_pipeline(data_path: str) -> PipelineDefault:
        """
        Get suggested static pipeline (which will be cached), typically loading the data plus design choices that we won't experiment with.
        :param data_path: path to original kits21 data (can be downloaded by KITS21.download())
        """
        static_pipeline = PipelineDefault("static",[
            # TODO: Should implement a decoder like kits? YES!
            (OpSkinLesionSampleIDDecode(), dict()),
            
            # Load Image
            (OpLoadRGBImage(data_path), dict(key_in="data.input.img_path", key_out="data.input.img")),

            # Normalize Image to range [0, 1]
            (OpNormalizeAgainstSelfImpl(), dict(key="data.input.img"))
        ])
        return static_pipeline

    
    @staticmethod
    def dynamic_pipeline() -> PipelineDefault:
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations. 
        """

        dynamic_pipeline = PipelineDefault("dynamic", [
                
                # Resize
                (OpResize2D(), dict(key="data.input.img", resize_to=(299,299)))

                # Padding

                # Augmentation



                # resize image to (110, 256, 256)
                # (OpLambda(func=partial(my_resize, resize_to=(110, 256, 256)))),

                # # Numpy to tensor
                # (OpToTensor(), kwargs_per_step_to_add=repeat_for)),
                
                # # affine transformation per slice but with the same arguments
                # (OpAugAffine2D(), dict(
                #     rotate=Uniform(-180.0,180.0),        
                #     scale=Uniform(0.8, 1.2),
                #     flip=(RandBool(0.5), RandBool(0.5)),
                #     translate=(RandInt(-15, 15), RandInt(-15, 15))
                # )),

                # # color augmentation - check if it is useful in CT images
                # (OpSample(OpAugColor()), dict(
                #     key="data.input.img",
                #     gamma=Uniform(0.8,1.2), 
                #     contrast=Uniform(0.9,1.1),
                #     add=Uniform(-0.01, 0.01)
                # )),

                # # add channel dimension -> [C=1, D, H, W]
                # (OpLambda(lambda x: x.unsqueeze(dim=0)), dict(key="data.input.img")),  
        ])
        return dynamic_pipeline


    @staticmethod
    def dataset(data_path: str, cache_dir: str, reset_cache: bool = False, num_workers:int = 10, sample_ids: Optional[Sequence[Hashable]] = None) -> DatasetDefault:
        """
        Get cached dataset
        :param data_path: path to store the original data
        :param cache_dir: path to store the cache
        :param reset_cache: set to True tp reset the cache
        :param num_workers: number of processes used for caching 
        :param sample_ids: dataset including the specified sample_ids or None for all the samples. sample_id is case_{id:05d} (for example case_00001 or case_00100).
        """
        train_data_path = os.path.join(data_path, 'ISIC2019/ISIC_2019_Training_Input')
        if sample_ids == None:
            sample_ids = SkinLesion.sample_ids(train_data_path)

        print("sample_ids' length:", len(sample_ids))

        static_pipeline = SkinLesion.static_pipeline(train_data_path)
        dynamic_pipeline = SkinLesion.dynamic_pipeline()

        # cacher = SamplesCacher(f'skin_lesion_cache_ver{SkinLesion.DATASET_VER}', 
        #     static_pipeline,
        #     [cache_dir], restart_cache=reset_cache, workers=num_workers)  

        my_dataset = DatasetDefault(sample_ids=sample_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
            cacher=None, #CHANGE    
        )

        my_dataset.create()
        print("my_dataset's length:", len(my_dataset))
        print(my_dataset[0])
        return my_dataset