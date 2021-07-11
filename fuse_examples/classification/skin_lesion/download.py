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


def download_and_extract_isic(root_data: str, year: str = '2016'):
    """
    Download images and metadata from ISIC challenge
    :param root_data: path where data should be located
    :param year: ISIC challenge year (2016 or 2017)
    """
    lgr = logging.getLogger('Fuse')

    if year == '2016':
        # 2016 - Train
        if not os.path.exists(os.path.join(root_data, 'data/ISIC2016_Training_Data')):
            lgr.info('Extract ISIC-2016 training data...')

            url = 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip'
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(os.path.join(root_data, 'data'))
            os.rename(os.path.join(root_data, 'data/ISBI2016_ISIC_Part3_Training_Data'),
                      os.path.join(root_data, 'data/ISIC2016_Training_Data'))

            lgr.info('Extracting ISIC-2016 training data: done')

        if not os.path.exists(os.path.join(root_data, 'data/ISIC2016_Training_GroundTruth.csv')):
            url = 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv'
            wget.download(url, os.path.join(root_data, 'data/ISIC2016_Training_GroundTruth.csv'))

        # 2016 - Test
        if not os.path.exists(os.path.join(root_data, 'data/ISIC2016_Test_Data')):
            lgr.info('Extract ISIC-2016 test data...')

            url = 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip'
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(os.path.join(root_data, 'data'))
            os.rename(os.path.join(root_data, 'data/ISBI2016_ISIC_Part3_Test_Data'),
                      os.path.join(root_data, 'data/ISIC2016_Test_Data'))

            lgr.info('Extracting ISIC-2016 test data: done')

        if not os.path.exists(os.path.join(root_data, 'data/ISIC2016_Test_GroundTruth.csv')):
            url = 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv'
            wget.download(url, os.path.join(root_data, 'data/ISIC2016_Test_GroundTruth.csv'))

    if year == '2017':
        # 2017 - Train
        if not os.path.exists(os.path.join(root_data, 'data/ISIC2017_Training_Data')):
            lgr.info('\nExtract ISIC-2017 training data... (this may take a few minutes)')

            url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip'
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(os.path.join(root_data, 'data'))
            os.rename(os.path.join(root_data, 'data/ISIC-2017_Training_Data'),
                      os.path.join(root_data, 'data/ISIC2017_Training_Data'))

            lgr.info('Extracting ISIC-2017 training data: done')

        if not os.path.exists(os.path.join(root_data, 'data/ISIC2017_Training_GroundTruth.csv')):
            url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part3_GroundTruth.csv'
            wget.download(url, os.path.join(root_data, 'data/ISIC2017_Training_GroundTruth.csv'))

        # 2017 - Validation
        if not os.path.exists(os.path.join(root_data, 'data/ISIC2017_Validation_Data')):
            lgr.info('\nExtract ISIC-2017 validation data... (this may take a few minutes)')

            url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip'
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(os.path.join(root_data, 'data'))
            os.rename(os.path.join(root_data, 'data/ISIC-2017_Validation_Data'),
                      os.path.join(root_data, 'data/ISIC2017_Validation_Data'))

            lgr.info('Extracting ISIC-2017 validation data: done')

        if not os.path.exists(os.path.join(root_data, 'data/ISIC2017_Validation_GroundTruth.csv')):
            url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part3_GroundTruth.csv'
            wget.download(url, os.path.join(root_data, 'data/ISIC2017_Validation_GroundTruth.csv'))

        # 2017 - Test
        if not os.path.exists(os.path.join(root_data, 'data/ISIC2017_Test_Data')):
            lgr.info('\nExtracting ISIC-2017 test data... (this may take a few minutes)')

            url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip'
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(os.path.join(root_data, 'data'))
            os.rename(os.path.join(root_data, 'data/ISIC-2017_Test_v2_Data'),
                      os.path.join(root_data, 'data/ISIC2017_Test_Data'))

            lgr.info('Extracting ISIC-2017 test data: done')

        if not os.path.exists(os.path.join(root_data, 'data/ISIC2017_Test_GroundTruth.csv')):
            url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv'
            wget.download(url, os.path.join(root_data, 'data/ISIC2017_Test_GroundTruth.csv'))



