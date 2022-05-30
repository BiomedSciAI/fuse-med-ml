
import os


from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool
from torch.utils.data.dataloader import DataLoader
from fuse.utils.ndict import NDict
import torch

from fuse.data import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data import PipelineDefault, OpSampleAndRepeat, OpToTensor, OpRepeat
from fuse.data.ops.op_base import OpBase
from fuse.data.ops.ops_aug_common import OpSample
from fuse.data.ops.ops_common import OpLambda
from fuseimg.data.ops.aug.color import OpAugColor
from fuseimg.data.ops.aug.geometry import OpAugAffine2D, OpAugSqueeze3Dto2D, OpAugUnsqueeze3DFrom2D
from fuseimg.data.ops.image_loader import OpLoadImage 
from fuseimg.data.ops.color import OpClip, OpToRange
import numpy as np
from fuse.data.utils.sample import get_sample_id
from typing import Hashable, List, Optional, Sequence, Tuple, Union
from functools import partial
import torch
from torch import Tensor
import pandas as pd
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault


class OpKnightSampleIDDecode(OpBase):
    '''
    decodes sample id into image and segmentation filename
    '''

    def __call__(self, sample_dict: NDict, op_id: Optional[str]) -> NDict:
        '''
        
        '''

        sid = get_sample_id(sample_dict)
        
        img_filename_key = 'data.input.img_path'
        sample_dict[img_filename_key] =   os.path.join(sid, 'imaging.nii.gz')

        seg_filename_key = 'data.gt.seg_path'
        sample_dict[seg_filename_key] = os.path.join(sid, 'aggregated_MAJ_seg.nii.gz')

        cols = ['case_id', 'age_at_nephrectomy', 'body_mass_index', 'gender', 'comorbidities', \
                'aua_risk_group', 'smoking_history', 'radiographic_size', 'last_preop_egfr']

        json_data = pd.read_json("/projects/msieve/MedicalSieve/PatientData/KNIGHT/knight/data/knight.json")[cols]
        row = json_data[json_data["case_id"]==sid].to_dict("records")[0]

        row['gender'] = int(row['gender'].lower() == 'female') #female:1 | male:0
        row["comorbidities"] = int(any(x for x in row["comorbidities"].values())) #if has any comorbidity it is set to 1
        row['smoking_history'] = ['never_smoked','previous_smoker','current_smoker'].index(row['smoking_history'])
        if row['last_preop_egfr'] is None or row['last_preop_egfr']['value'] is None:
            row['last_preop_egfr'] = 77 # median value 
        elif row['last_preop_egfr']['value'] in ('>=90', '>90'):
            row['last_preop_egfr'] = 90
        else:
            row['last_preop_egfr'] = row['last_preop_egfr']['value']

        if row['radiographic_size'] is None:
            row['radiographic_size'] = 4.1 # this is the median value on the training set

        sample_dict["data.gt.gt_global.task_1_label"] = int(row["aua_risk_group"] in ['high_risk', 'very_high_risk'])
        sample_dict["data.gt.gt_global.task_2_label"] = ['benign','low_risk','intermediate_risk','high_risk', 'very_high_risk'].index(row["aua_risk_group"])

        sample_dict["data.input.clinical"] = row
        return sample_dict

def aug_op_random_crop_and_pad(aug_input: Tensor,
                           out_size: Tuple,
                           fill: int = 0,
                           centralize: bool=False ) -> Tensor:
    """
    random crop to certain size. if the image is smaller than the size then its padded.
    :param aug_input: The tensor to augment
    :param out_size: shape of the output
    :return: the augmented tensor
    """
    assert len(aug_input.shape) == len(out_size)
    depth, height, width = aug_input.shape #input is in the form [D,H,W]

    aug_tensor = torch.full(out_size, fill, dtype=torch.float32)

    if depth > out_size[0]:
        crop_start = RandInt(0, depth - out_size[0]).sample()
        if centralize:
            crop_start = round((depth - out_size[0])/2)
        aug_input = aug_input[crop_start:crop_start+out_size[0] , :,:]
    if height > out_size[1]:
        crop_start = RandInt(0, height - out_size[1]).sample()
        if centralize:
            crop_start = round((height - out_size[1])/2)
        aug_input = aug_input[:, crop_start:crop_start+out_size[1],:]
    if width > out_size[2]:
        crop_start = RandInt(0, width - out_size[2]).sample()
        if centralize:
            crop_start = round((width - out_size[2])/2)
        aug_input = aug_input[:,:,crop_start:crop_start+out_size[2]]

    aug_tensor[:depth,:height,:width] = aug_input

    return aug_tensor

class OpPrepare_Clinical(OpBase):

    def __call__(self, sample_dict: NDict, op_id: Optional[str]) -> NDict:
        age = sample_dict['data.input.clinical.age_at_nephrectomy']
        if age!=None and age > 0 and age < 120:
            age = np.array(age / 120.0).reshape(-1)
        else:
            age = np.array(-1.0).reshape(-1)
        
        bmi = sample_dict['data.input.clinical.body_mass_index']
        if bmi!=None and bmi > 10 and bmi < 100:
            bmi = np.array(bmi / 50.0).reshape(-1)
        else:
            bmi = np.array(-1.0).reshape(-1)

        radiographic_size = sample_dict['data.input.clinical.radiographic_size']
        if radiographic_size!=None and radiographic_size > 0 and radiographic_size < 50:
            radiographic_size = np.array(radiographic_size / 15.0).reshape(-1)
        else:
            radiographic_size = np.array(-1.0).reshape(-1)
        
        preop_egfr = sample_dict['data.input.clinical.last_preop_egfr']
        if preop_egfr!=None and preop_egfr > 0 and preop_egfr < 200:
            preop_egfr = np.array(preop_egfr / 90.0).reshape(-1)
        else:
            preop_egfr = np.array(-1.0).reshape(-1)
        # turn categorical features into one hot vectors
        gender = sample_dict['data.input.clinical.gender']
        gender_one_hot = np.zeros(len(GENDER_INDEX))
        if gender in GENDER_INDEX.values():
            gender_one_hot[gender] = 1

        comorbidities = sample_dict['data.input.clinical.comorbidities']
        comorbidities_one_hot = np.zeros(len(COMORBIDITIES_INDEX))
        if comorbidities in COMORBIDITIES_INDEX.values():
            comorbidities_one_hot[comorbidities] = 1
        
        smoking_history = sample_dict['data.input.clinical.smoking_history']
        smoking_history_one_hot = np.zeros(len(SMOKE_HISTORY_INDEX))
        if smoking_history in SMOKE_HISTORY_INDEX.values():
            smoking_history_one_hot[smoking_history] = 1
        

        clinical_encoding = np.concatenate((age, bmi, radiographic_size, preop_egfr, gender_one_hot, comorbidities_one_hot, smoking_history_one_hot), axis=0, dtype=np.float32)
        sample_dict["data.input.clinical.all"] = clinical_encoding
        return sample_dict

def knight_dataset(data_dir: str = 'data', cache_dir: str = 'cache', split: dict = None, \
        reset_cache: bool = False, \
        rand_gen = None, batch_size=8, resize_to=(256,256,110), task_num=1, \
        target_name='data.gt.gt_global.task_1_label', num_classes=2, only_labels=False):
    

    static_pipeline = PipelineDefault("static", [
        # decoding sample ID
        (OpKnightSampleIDDecode(), dict()), # will save image and seg path to "data.input.img_path", "data.gt.seg_path" 
        
        # loading data
        (OpLoadImage(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
        # (OpLoadImage(data_dir), dict(key_in="data.gt.seg_path", key_out="data.gt.seg", format="nib")),
        
        
        # fixed image normalization
        (OpClip(), dict(key="data.input.img", clip=(-62, 301))),
        (OpLambda(lambda x: (x - 104.0)/75.3 ), dict(key="data.input.img")), #kits normalization
        # (OpToRange(), dict(key="data.input.img", from_range=(-500, 500), to_range=(0, 1))),
        
        # transposing so the depth channel will be first
        (OpLambda(lambda x: np.moveaxis(x, -1, 0)), dict(key="data.input.img")), # convert image from shape [H, W, D] to shape [D, H, W] 
        (OpPrepare_Clinical(), dict()), #process clinical data

    ])

    dynamic_pipeline = PipelineDefault("dynamic", [
                
        # resize image to (110, 256, 256)
        # (OpLambda(func=partial(my_resize, resize_to=(110, 256, 256))), dict(key="data.input.img")),

        # Numpy to tensor
        (OpToTensor(), dict(key="data.input.img")),
        (OpToTensor(), dict(key="data.input.clinical.all")),


        # (OpAugSqueeze3Dto2D(), dict(key="data.input.img")),
        # # affine transformation per slice but with the same arguments
        # (OpAugAffine2D() , dict(
        #     key="data.input.img",
        #     rotate=Uniform(-180.0,180.0),        
        #     scale=Uniform(0.8, 1.2),
        #     flip=(RandBool(0.5), RandBool(0.5)),
        #     translate=(RandInt(-15, 15), RandInt(-15, 15))
        # )),
        # (OpAugUnsqueeze3DFrom2D(), dict(key="data.input.img")),


        # color augmentation - check if it is useful in CT images
        # (OpSample(OpAugColor()), dict(
        #     key="data.input.img",
        #     gamma=Uniform(0.8,1.2), 
        #     contrast=Uniform(0.9,1.1),
        #     add=Uniform(-0.01, 0.01)
        # )),
        (OpLambda(lambda x: aug_op_random_crop_and_pad(x, resize_to, centralize=False)), dict(key="data.input.img")),
        # add channel dimension -> [C=1, D, H, W]
        (OpLambda(lambda x: x.unsqueeze(dim=0)), dict(key="data.input.img")),  
    ]) 
       
    
    if 'train' in split:
        image_dir = data_dir
        json_filepath = os.path.join(image_dir, 'knight.json')
        json_filename = os.path.join(image_dir, 'knight.json')
        clinical_data = pd.read_json(json_filename)
        # gt_processors = {
        #     'gt_global': KiCGTProcessor(json_filename=json_filepath, columns_to_tensor={'task_1_label':torch.long, 'task_2_label':torch.long})
        # }
    else: # split can contain BOTH 'train' and 'val', or JUST 'test'
        image_dir = os.path.join(data_dir, 'images')
        json_filepath = os.path.join(data_dir, 'features.json')
        # if only_labels:
        #     json_labels_filepath = os.path.join(data_dir, 'knight_test_labels.json') 
        #     gt_processors = {
        #         'gt_global': KiCGTProcessor(json_filename=json_labels_filepath, columns_to_tensor={'task_1_label':torch.long, 'task_2_label':torch.long}, test_labels=True)
        #     }
        # else:
        #     gt_processors = {}

    # if only_labels:
    #     # just labels - no need to load and process input
    #     input_processors = {}
    #     post_processing_func=None
    # else:
    #     # we use the same processor for the clinical data and ground truth, since both are in the .csv file
    #     # need to make sure to discard the label column from the data when using it as input
    #     input_processors = {
    #         'image': KiTSBasicInputProcessor(input_data=image_dir, resize_to=resize_to),
    #         'clinical': KiCClinicalProcessor(json_filename=json_filepath)
    #     }
    #     post_processing_func=prepare_clinical


       # Create dataset
    if 'train' in split:
        train_cacher = SamplesCacher("train_cache", 
        static_pipeline,
        cache_dirs=[f"{cache_dir}/train"], restart_cache=reset_cache)

        train_dataset = DatasetDefault(sample_ids=split['train'],
        static_pipeline=static_pipeline,
        dynamic_pipeline=dynamic_pipeline,
        cacher=train_cacher)

        print(f'- Load and cache data:')
        train_dataset.create()
    
        print(f'- Load and cache data: Done')

        ## Create sampler
        print(f'- Create sampler:')
        sampler = BatchSamplerDefault(dataset=train_dataset,
                                        balanced_class_name=target_name,
                                        num_balanced_classes=num_classes,
                                        batch_size=batch_size,
                                        balanced_class_weights=[1.0/num_classes]*num_classes if task_num==2 else None)
                                                              

        print(f'- Create sampler: Done')

        ## Create dataloader
        train_dataloader = DataLoader(dataset=train_dataset,
                                    shuffle=False, drop_last=False,
                                    batch_sampler=sampler, collate_fn=CollateDefault(),
                                    num_workers=8, generator=rand_gen)
        print(f'Train Data: Done', {'attrs': 'bold'})

        #### Validation data
        print(f'Validation Data:', {'attrs': 'bold'})

        val_cacher = SamplesCacher("val_cache", 
            static_pipeline,
            cache_dirs=[f"{cache_dir}/val"], restart_cache=reset_cache)
        ## Create dataset
        validation_dataset = DatasetDefault(sample_ids=split['val'],
        static_pipeline=static_pipeline,
        dynamic_pipeline=dynamic_pipeline,
        cacher=val_cacher)

        print(f'- Load and cache data:')
        validation_dataset.create()
        print(f'- Load and cache data: Done')

        ## Create dataloader
        validation_dataloader = DataLoader(dataset=validation_dataset,
                                        shuffle=False,
                                        drop_last=False,
                                        batch_sampler=None,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        collate_fn=CollateDefault(),
                                        generator=rand_gen)
        print(f'Validation Data: Done', {'attrs': 'bold'})
        test_dataloader = test_dataset = None
    else: # test only
        #### Test data
        print(f'Test Data:', {'attrs': 'bold'})

        ## Create dataset
        test_dataset = DatasetDefault(sample_ids=split['test'],
        static_pipeline=static_pipeline,
        dynamic_pipeline=dynamic_pipeline,)

        print(f'- Load and cache data:')
        test_dataset.create(pool_type='thread')  # use ThreadPool to create this dataset, to avoid cv2 problems in multithreading
        print(f'- Load and cache data: Done')

        ## Create dataloader
        test_dataloader = DataLoader(dataset=test_dataset,
                                        shuffle=False,
                                        drop_last=False,
                                        batch_sampler=None,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        collate_fn=CollateDefault(),
                                        generator=rand_gen)
        print(f'Test Data: Done', {'attrs': 'bold'})
        train_dataloader = train_dataset = validation_dataloader = validation_dataset = None
    return train_dataloader, validation_dataloader, test_dataloader, \
            train_dataset, validation_dataset, test_dataset


GENDER_INDEX = {
    'male': 0,
    'female': 1
}
COMORBIDITIES_INDEX = {
    'no comorbidities': 0,
    'comorbidities exist': 1
}
SMOKE_HISTORY_INDEX = {
    'never smoked': 0,
    'previous smoker': 1,
    'current smoker': 2
}
