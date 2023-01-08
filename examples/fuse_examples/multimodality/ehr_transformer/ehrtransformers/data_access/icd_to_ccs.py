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

from os import environ
import os
import pandas as pd

# from ibm_db import connect
# from ibm_db_dbi import Connection
import json


def get_icd_ccs_dict(icd_ccs_map_df):
    icd_ccs_dict = icd_ccs_map_df.set_index("CODE")["CCS_CATEGORY"].to_dict()
    repeat_codes_dict = get_repeat_codes_dict(icd_ccs_map_df)
    icd_ccs_dict.update(repeat_codes_dict)
    return icd_ccs_dict


def get_repeat_codes_dict(icd_ccs_map_df, icd_to_use="icd9"):
    repeat_codes = icd_ccs_map_df[icd_ccs_map_df.duplicated("CODE")]
    repeat_codes = repeat_codes[repeat_codes.ICD_VERSION == icd_to_use]
    return repeat_codes.set_index("CODE")["CCS_CATEGORY"].to_dict()


def read_icd_ccs_dict(json_filepath=None):
    if json_filepath is None:
        log_path = os.path.abspath(__file__)
        code_path = os.path.dirname(os.path.dirname(os.path.dirname(log_path)))
        json_filepath = os.path.join(code_path, "ehrtransformers/data_access/icd_to_ccs.json")

    with open(json_filepath) as f:
        tmp = json.load(f)
        return tmp
