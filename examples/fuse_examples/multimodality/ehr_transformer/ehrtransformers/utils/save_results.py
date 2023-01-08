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

from ehrtransformers.utils.common import load_pkl
from ehrtransformers.configs.config import get_config
from sys import argv
from ehrtransformers.utils.log_utils import get_output_dir
import pickle
import gzip
import os
from datetime import datetime
import numpy as np
import pandas as pd


def create_output_df(df_out, df_input, naming_conventions):
    """
    Combines information from train/inference input and output dataframes for further analysis
    :param df_out: dataframe containing outputs of a train/inference run of a model
    :param df_input: dataframe (originally used as input for model training/validation/test) containing additional patient information
    :param naming_conventions: a dictionary containing relevant column names in the input dfs:
        {'age_key': 'AGE',
        'age_month_key': 'AGE_MON',
        'date_key': 'ADMDATE',
        'index_date_key': 'INDDATE',
        'adm_date_key': 'ADMDATE',
        'svc_date_key': 'SVCDATE',
        'date_birth_key': 'DOBYR',
        'diagnosis_vec_key': 'DX',
        'gender_key': 'SEX',
        'label_key': 'label',
        'outcome_key': 'RC',
        'event_key': 'EVENTS',
        'treatment_event_key':
        'TREATMENT_EVENTS',
        'patient_id_key': 'ENROLID',
        'separator_str': 'SEP',
        'dxver_key': 'DXVER',
        'healthy_val': 0,
        'sick_val': 1,
        'split_key':   'split',
        'fold_key': 'fold',
        'next_visit_key': 'DXNEXTVIS'}
    :return:
    """
    df_res = df_out.copy(deep=True)
    patid_key = "patid"
    patid_key_in_db = naming_conventions["patient_id_key"]
    date_key_in_db = naming_conventions["date_key"]
    age_key = naming_conventions["age_key"]
    gender_key = naming_conventions["gender_key"]
    disease_key = naming_conventions["outcome_key"]
    vis_date_key = "ADMDATE"
    encoding_key = "encodings"
    date_format_string = "%Y%m%d"
    rename_dict = {
        "data.patid": patid_key,
        "data.vis_date": vis_date_key,
        "model.backbone_features": encoding_key,
    }
    df_res.rename(columns=rename_dict, inplace=True)
    df_res[patid_key] = df_res[patid_key].apply(lambda x: x[0])
    df_res[vis_date_key] = df_res[vis_date_key].apply(lambda x: datetime.strptime(str(x[0]), date_format_string).date())
    tmp = df_input[[patid_key_in_db, date_key_in_db, age_key, gender_key, disease_key]]
    rename_dict = {
        patid_key_in_db: patid_key,
        date_key_in_db: vis_date_key,
    }
    tmp.rename(columns=rename_dict, inplace=True)
    if isinstance(tmp[vis_date_key][0], datetime):
        tmp[vis_date_key] = tmp[vis_date_key].apply(lambda x: x.date())
    df_res = df_res.merge(tmp, how="inner", on=[patid_key, vis_date_key])
    df_res[age_key] = df_res[age_key].apply(lambda x: x[-1])
    # df_res.drop('descriptor', inplace=True)
    return df_res


def save_translated_outputs(global_params, file_config, fnames, naming_conventions, index=0):
    """
    The function translates BEHRT fuse output files to a utils format (dataframe with certain columns).
    It also adds some information from the relevant input DF (train/val/test GT/statistics).
    Note: Since we're only given inference file names, we need to infer which data set (train/val/test) to use
    from the filename. This is not optimal, and should be changed. TODO:
    :param global_params:
    :param file_config:
    :param fnames: list of inference files
    :param naming_conventions: a dictionary containing relevant column names in the input dfs:
        {'age_key': 'AGE',
        'age_month_key': 'AGE_MON',
        'date_key': 'ADMDATE',
        'index_date_key': 'INDDATE',
        'adm_date_key': 'ADMDATE',
        'svc_date_key': 'SVCDATE',
        'date_birth_key': 'DOBYR',
        'diagnosis_vec_key': 'DX',
        'gender_key': 'SEX',
        'label_key': 'label',
        'outcome_key': 'RC',
        'event_key': 'EVENTS',
        'treatment_event_key':
        'TREATMENT_EVENTS',
        'patient_id_key': 'ENROLID',
        'separator_str': 'SEP',
        'dxver_key': 'DXVER',
        'healthy_val': 0,
        'sick_val': 1,
        'split_key':   'split',
        'fold_key': 'fold',
        'next_visit_key': 'DXNEXTVIS'}
    :param index: index of the run_ dir. if >0, it uses the absolute index. If <=0 - it uses the relative index (
    0 - last run, -n: n-before-last)
    :return:
    """
    main_output_dir = get_output_dir(global_params, index=index)
    infer_model_dir = os.path.join(main_output_dir, "infer_dir")

    for fname in fnames:
        fpath = os.path.join(infer_model_dir, fname + ".gz")
        dset = "val"
        if "train" in fname:
            dset = "train"
        if "test" in fname:
            dset = "test"
        if os.path.exists(fpath):
            out_fpath = os.path.join(infer_model_dir, fname + "_translated.gz")
            with gzip.open(fpath, "rb") as f:
                df_out = pickle.load(f)
            with open(file_config[dset] + ".pkl", "rb") as f_val:
                data_val = pickle.load(f_val)
                df_input = data_val["data_df"]
            out_df = create_output_df(df_out, df_input, naming_conventions)
            with gzip.open(out_fpath, "wb") as f_out:
                pickle.dump(out_df, f_out)
            out_df.to_csv(out_fpath.replace(".gz", ".csv"), index=False)


if __name__ == "__main__":
    if len(argv) > 1:
        config_file_path = argv[1]
    else:
        config_file_path = None

    (global_params, file_config, model_config, optim_config, data_config, naming_conventions,) = get_config(
        config_file_path
    )

    fnames = ["validation_set_infer_best", "validation_set_infer_last", "train_set_infer_best", "train_set_infer_last"]

    save_translated_outputs(global_params, file_config, fnames=fnames, index=133)
