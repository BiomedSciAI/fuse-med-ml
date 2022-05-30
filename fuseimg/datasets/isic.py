import os
from sqlite3 import SQLITE_CREATE_TEMP_TABLE
from zipfile import ZipFile
import io
import wget
import logging
from typing import Hashable, Optional, Sequence, List
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import torch

from fuse.data import DatasetDefault
from fuse.data.ops.ops_cast import OpToNumpy, OpToTensor, OpOneHotToNumber
from fuse.data.utils.sample import get_sample_id
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.op_base import OpBase
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.ops_aug_common import OpSample
from fuse.data.ops.ops_read import OpReadDataframe, OpReadLabelsFromDF
from fuse.utils import NDict

from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.color import OpNormalizeAgainstSelf, OpPad
from fuseimg.data.ops.aug.color import OpAugColor, OpAugGaussian
from fuseimg.data.ops.aug.geometry import OpResizeTo, OpAugAffine2D
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool


class OpISICSampleIDDecode(OpBase):

    def __call__(self, sample_dict: NDict) -> NDict:
        """
        decodes sample id into image file name
        """
        sid = get_sample_id(sample_dict)

        img_filename_key = 'data.input.img_path'
        sample_dict[img_filename_key] =   sid + '.jpg'

        return sample_dict


class ISIC:
    """
    ISIC 2019 challenge to classify dermoscopic images and clinical data among nine different diagnostic categories.
    """
    # bump whenever the static pipeline modified
    DATASET_VER = 0
    
    def __init__(self,
                 data_path: str,
                 cache_path: str,
                 val_portion: float=0.3, 
                 partition_file: Optional[str]=None) -> None:
        """
        
        :param data_path:
        :param cache_path:
        :param val_portion:
        :param override_partition:
        :param partition_file:
        """
        self.data_path = data_path
        self.cache_path = cache_path
        self.val_portion = val_portion
        self.partition_file = partition_file
        self._downloaded = False

        if partition_file is None:
            self.partition_file = os.path.join(data_path, 'ISIC2019/partition.pickle')


    def download(self) -> None:
        """
        Download images and metadata from ISIC challenge.
        Doesn't download again if data exists.

        """
        lgr = logging.getLogger('Fuse')
    
        path = os.path.join(self.data_path, 'ISIC2019/ISIC_2019_Training_Input')
        print(f"Training Input Path: {os.path.abspath(path)}")
        if not os.path.exists(path):
            lgr.info('\nExtract ISIC-2019 training input ... (this may take a few minutes)')

            url = 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip'
            wget.download(url, ".")
            
            with ZipFile("ISIC_2019_Training_Input.zip", 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(path=os.path.join(self.data_path, 'ISIC2019'))

            lgr.info('Extracting ISIC-2019 training input: done')

        path = os.path.join(self.data_path, 'ISIC2019/ISIC_2019_Training_GroundTruth.csv')

        if not os.path.exists(path):
            lgr.info('\nExtract ISIC-2019 training gt ... (this may take a few minutes)')

            url = 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv'
            wget.download(url, path)

            lgr.info('Extracting ISIC-2019 training gt: done')
        
        self._downloaded = True

    def sample_ids(self, size: Optional[int] = None) -> List[str]:
        """
        Gets the samples ids in trainset.
        If size is not None, return only the first 'size' ids.
        """
        images_path = os.path.join(self.data_path, 'ISIC2019/ISIC_2019_Training_Input')
        
        samples = [f.split('.')[0] for f in os.listdir(images_path) if f.split('.')[-1] == 'jpg']

        # Take only size elements
        # TODO: input validation
        if size is not None:
            samples = samples[-1 * size:]

        return samples

    @staticmethod
    def static_pipeline(data_path: str) -> PipelineDefault:
        """
        Get suggested static pipeline (which will be cached), typically loading the data plus design choices that we won't experiment with.
        :param data_path: path to original kits21 data (can be downloaded by KITS21.download())
        """
        static_pipeline = PipelineDefault("static",[
            # Decoding sample ID
            (OpISICSampleIDDecode(), dict()),
            
            # Load Image
            (OpLoadImage(data_path), dict(key_in="data.input.img_path", key_out="data.input.img")),
            
            # To Tensor debugging
            # (OpToTensor(), dict(key="data.input.img", dtype=torch.int)),

            # Normalize Image to range [0, 1]
            (OpNormalizeAgainstSelf(), dict(key="data.input.img")),

            # Cast to numpy array for caching purposes
            (OpToNumpy(), dict(key="data.input.img")),

            # Read labels
            (OpReadLabelsFromDF(data_filename=os.path.join(data_path, '../ISIC_2019_Training_GroundTruth.csv'), key_column="image"), dict()),
        ])
        return static_pipeline

    @staticmethod
    def dynamic_pipeline() -> PipelineDefault:
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations. 
        """

        dynamic_pipeline = PipelineDefault("dynamic", [
                # Resize images to 3x300x300
                (OpResizeTo(), dict(key="data.input.img", resize_to=[3, 500, 500])),
                
                # Cast to Tensor
                (OpToTensor(), dict(key="data.input.img")),

                # Padding
                # (OpPad(), dict(key="data.input.img", padding=1)),

                # # Augmentation                
                # (OpSample(OpAugAffine2D()), dict(
                #     key="data.input.img",
                #     rotate=Uniform(-180.0,180.0),        
                #     scale=Uniform(0.9, 1.1),
                #     flip=(RandBool(0.5), RandBool(0.5)),
                #     translate=(RandInt(-50, 50), RandInt(-50, 50))
                # )),

                # # Color augmentation
                # (OpSample(OpAugColor()), dict(
                #     key="data.input.img",
                #     gamma=Uniform(0.9,1.1), 
                #     contrast=Uniform(0.85,1.15),
                #     add=Uniform(-0.06, 0.06),
                #     mul = Uniform(0.95, 1.05)
                # )),

                # # Gaussian noise
                # (OpAugGaussian(), dict(key="data.input.img", std=0.03)),

                (OpOneHotToNumber(num_classes=9), dict(key="data.label"))
        ])

        return dynamic_pipeline

    def dataset(self,
                train: bool=True,
                size: Optional[int] = None,
                reset_cache: bool = False, 
                num_workers:int = 10,
                samples_ids: Optional[Sequence[Hashable]] = None,
                override_partition: bool = True) -> DatasetDefault:
        """
        Get cached dataset
        :param data_path: path to store the original data
        :param cache_dir: path to store the cache
        :param reset_cache: set to True to reset the cache
        :param num_workers: number of processes used for caching 
        :param sample_ids: dataset including the specified sample_ids or None for all the samples.
        """
        train_data_path = os.path.join(self.data_path, 'ISIC2019/ISIC_2019_Training_Input')
        labels_path = os.path.join(self.data_path, 'ISIC2019/ISIC_2019_Training_GroundTruth.csv')
        labels_df = pd.read_csv(labels_path)
        self._labels_df = labels_df #.drop(axis=1, labels=['UNK']) # UNK label has 0 instances TODO

        if samples_ids is None:
            samples_ids = self.sample_ids(size)

        if train: 
            print("partition's path exist?", os.path.exists(self.partition_file))
            if override_partition or not os.path.exists(self.partition_file):
                train_samples, val_samples = train_test_split(samples_ids, test_size=self.val_portion, random_state=42)
                splits = {'train': train_samples, 'val': val_samples}

                with open(self.partition_file, "wb") as pickle_out:
                    pickle.dump(splits, pickle_out)
                    out_samples_ids = splits['train']
                    print("Dumped pickle!")

            else:
                # read from a previous split to evaluate on the same partition
                with open(self.partition_file, "rb") as splits:
                    repartition = pickle.load(splits)
                    out_samples_ids = repartition['train']

        else:
            # return validation set according to the partition
            with open(self.partition_file, "rb") as splits:
                repartition = pickle.load(splits)
                out_samples_ids = repartition['val']
        
        static_pipeline = ISIC.static_pipeline(train_data_path)
        dynamic_pipeline = ISIC.dynamic_pipeline()

        cacher = SamplesCacher(f'isic_cache_ver{ISIC.DATASET_VER}', 
            static_pipeline,
            [self.cache_path], restart_cache=reset_cache, workers=num_workers)  

        my_dataset = DatasetDefault(sample_ids=out_samples_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
            cacher=cacher
        )

        my_dataset.create()
        return my_dataset
