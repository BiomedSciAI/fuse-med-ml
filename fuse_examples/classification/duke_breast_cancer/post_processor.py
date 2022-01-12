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

from typing import Dict
import torch
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
import numpy as np


def post_processing(batch_dict: Dict, label: str, is_concat_features_to_input: bool = False,
                ) -> None:
    """
    post_processing updates batch_dict on the post processing phase
    This function :
    1. Defines the classification label base on label input und update the ground_truth field in batch_dict
    2. Extracts the relevant tubular parameters from add_data field in the batch_dict and
       creates data.clinical_features
    :param batch_dict:
    :param label: label to use as classification label
    :param is_concat_features_to_input - if True, concat the  data.clinical_features to the input channels
    :return: updated batch_dict
    """

    # select input channel
    input_tensor = FuseUtilsHierarchicalDict.get(batch_dict, 'data.input')
    FuseUtilsHierarchicalDict.set(batch_dict, 'data.input', input_tensor[[0],:,:,:])
    clinical_features = FuseUtilsHierarchicalDict.get(batch_dict, 'data.add_data')



    # select label
    mylabel = label
    FuseUtilsHierarchicalDict.set(batch_dict, 'data.filter', False)

    if mylabel == 'ispCR':
        features_to_use = ['Skin Invovlement', 'Tumor Size US', 'Tumor Size MG', 'Field of View', 'Contrast Bolus Volume',
                           'Race','Manufacturer','Slice Thickness']

        ispCR = clinical_features['Near pCR Strict'] + 0
        if ispCR>2:
            FuseUtilsHierarchicalDict.set(batch_dict, 'data.filter', True)
            return
        elif ispCR==0 or ispCR==2:
            ispCR = 0
        else:
            ispCR = 1
        label_tensor = torch.tensor(ispCR, dtype = torch.int64)
        FuseUtilsHierarchicalDict.set(batch_dict, 'data.ground_truth', label_tensor)


    if mylabel == 'Staging Tumor Size':
        features_to_use = ['Skin Invovlement', 'Tumor Size US', 'Tumor Size MG', 'Field of View', 'Contrast Bolus Volume',
                           'Race','Manufacturer','Slice Thickness']

        TumorSize = clinical_features['Staging Tumor Size'] + 0
        if TumorSize>1:
            TumorSize=1
        else:
            TumorSize=0
        label_tensor = torch.tensor(TumorSize, dtype = torch.int64)
        FuseUtilsHierarchicalDict.set(batch_dict, 'data.ground_truth', label_tensor)

    if mylabel == 'Histology Type':
        features_to_use = ['Skin Invovlement', 'Tumor Size US', 'Tumor Size MG', 'Field of View', 'Contrast Bolus Volume',
                           'Race','Multicentric','Manufacturer','Slice Thickness']
        type = clinical_features['Histologic type'] + 0
        if type==1:
            type=0
        elif type==10:
            type=1
        else:
            FuseUtilsHierarchicalDict.set(batch_dict, 'data.filter', True)

        label_tensor = torch.tensor(type, dtype = torch.int64)
        FuseUtilsHierarchicalDict.set(batch_dict, 'data.ground_truth', label_tensor)


    if mylabel == 'is High Tumor Grade Total':
        features_to_use = ['Breast Density MG','PR','HER2','ER']
        grade = clinical_features['Tumor Grade Total'] + 0
        if grade>=7:
            grade = 1
        else:
            grade = 0

        label_tensor = torch.tensor(grade, dtype = torch.int64)
        FuseUtilsHierarchicalDict.set(batch_dict, 'data.ground_truth', label_tensor)

    if mylabel == 'Recurrence':
        features_to_use = []
        grade = clinical_features['Recurrence'] + 0

        label_tensor = torch.tensor(grade, dtype = torch.int64)
        FuseUtilsHierarchicalDict.set(batch_dict, 'data.ground_truth', label_tensor)


    # add clinical

    clinical_features_to_use = torch.tensor([float(clinical_features[feature]) for feature in features_to_use],dtype = torch.float32)
    FuseUtilsHierarchicalDict.set(batch_dict, 'data.clinical_features', clinical_features_to_use)

    if is_concat_features_to_input:
    # select input channel
        input_tensor = FuseUtilsHierarchicalDict.get(batch_dict, 'data.input')
        input_shape = input_tensor.shape
        for feature in clinical_features_to_use:
            input_tensor = torch.cat((input_tensor,feature.repeat(input_shape[1],input_shape[2],input_shape[3]).unsqueeze(0)),dim=0)
        FuseUtilsHierarchicalDict.set(batch_dict, 'data.input', input_tensor)