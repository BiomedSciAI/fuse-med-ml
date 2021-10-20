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

import os
import requests
import zipfile
import io
import wget

import logging


def download_and_extract_isic(root_data: str = 'data'):
    """
    Download images and metadata from ISIC challenge
    :param root_data: path where data should be located
    """
    lgr = logging.getLogger('Fuse')

    
    path = os.path.join(root_data, 'ISIC2019/ISIC_2019_Training_Input')
    print(f"Training Input Path: {os.path.abspath(path)}")
    if not os.path.exists(path):
        lgr.info('\nExtract ISIC-2019 training input ... (this may take a few minutes)')

        url = 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip'
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(os.path.join(root_data, 'ISIC2019'))

        lgr.info('Extracting ISIC-2019 training input: done')

    path = os.path.join(root_data, 'ISIC2019/ISIC_2019_Training_Metadata.csv')
    print(f"Training Metadata Path: {os.path.abspath(path)}")
    if not os.path.exists(path):
        lgr.info('\nExtract ISIC-2019 training metadata ... (this may take a few minutes)')

        url = 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv'
        wget.download(url, path)

        lgr.info('Extracting ISIC-2019 training metadata: done')

    path = os.path.join(root_data, 'ISIC2019/ISIC_2019_Training_GroundTruth.csv')
    print(f"Training Metadata Path: {os.path.abspath(path)}")
    if not os.path.exists(path):
        lgr.info('\nExtract ISIC-2019 training gt ... (this may take a few minutes)')

        url = 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv'
        wget.download(url, path)

        lgr.info('Extracting ISIC-2019 training gt: done')



