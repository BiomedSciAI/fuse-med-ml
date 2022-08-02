
from typing import Optional

import numpy as np
import pandas as pd

from fuse.utils import NDict


def get_samples_for_cohort(cohort_config: NDict, clinical_data_file:str, seed:Optional[int]=222, lgr=None):

    def write_log_info(s):
        if lgr is not None:
            lgr.info(s)

    df = pd.read_csv(clinical_data_file)

    filter_out_groups = cohort_config.get("filter_out")
    if filter_out_groups is not None:
        filter_out = np.zeros(df.shape[0], dtype=bool)
        for group_id in filter_out_groups:
            group_filter = get_group_filter(group_id, df)
            filter_out |= group_filter
    else:
        filter_out = None

    max_group_size = cohort_config[ 'max_group_size']
    max_group_size = None if max_group_size <= 0 else max_group_size
    np.random.seed(seed)

    selected = np.zeros(df.shape[0], dtype=bool)
    group_ids = cohort_config['group_ids']

    for group_id in group_ids:
        group_filter = get_group_filter(group_id, df)
        group_size = group_filter.sum()

        write_log_info(f'{group_id} size={group_size}')
        if filter_out is not None:
            group_filter &= ~filter_out
            group_size = group_filter.sum()
            write_log_info(f'{group_id} size={group_size} after filtering')

        if max_group_size is not None and group_size > max_group_size:
            all_indexes = np.where(group_filter)[0]
            rand_perm = np.random.permutation(group_size)
            n_remove = group_size -max_group_size
            indexes_to_remove = all_indexes[rand_perm[:n_remove]]
            assert np.all(group_filter[indexes_to_remove])
            group_filter[indexes_to_remove] = False
            assert np.sum(group_filter) == max_group_size
            write_log_info( f"{group_id} size: {group_size} => {max_group_size}, First removed index= {indexes_to_remove[0]}")

        selected |= group_filter
    print("cohort size=", np.sum(selected))
    return df['file_pattern'].values[selected].tolist()



def get_group_filter(group_id, df):
    if group_id == 'all':
        return np.ones(df.shape[0], dtype=bool)
    # sex
    if group_id == 'men':
        return df['is female'] == 0
    if group_id == 'women':
        return df['is female'] == 1

    # neoplasms (malignant, in situ, benign)
    if group_id == 'neoplasms':
        return df['preindex neoplasms'] >0
    if group_id == 'no_neoplasms':
        return df['preindex neoplasms'] == 0

    # malignant
    if group_id == 'cancer':
        return df['preindex cancer'] >0
    if group_id == 'no_cancer':
        return df['preindex cancer'] == 0

    # maligant - male genital
    if group_id == 'cancer_male_genital':
        return df['blocks preindex C60-C63 Malignant neoplasms of male genital organs'] > 0
    # malignent - urinary tract (covers kidney)
    if group_id == 'cancer_urinary_tract':
        return df[ 'blocks preindex C64-C68 Malignant neoplasms of urinary tract'] > 0
    # malignent - prostate
    if group_id == 'cancer_prostate':
        return df['preindex C61 Malignant neoplasm of prostate'] > 0
    # malignent - kidney
    if group_id == 'cancer_kidney':
        return df['preindex C64 Malignant neoplasm of kidney, except renal pelvis'] > 0

    # prostatectomy
    if group_id == 'prostatectomy':
        return df['preindex prostatectomy'] > 0
    if group_id == 'no_prostatectomy':
        return df['preindex prostatectomy'] == 0


    raise NotImplementedError(group_id)
def get_class_names(label_type:str):
    if label_type == "classification":
        class_names = ["Male", "Female","Male-prostate-excision"]
    elif label_type == "is female":
        class_names = ["Male", "Female"]
    elif label_type == "preindex prostatectomy":
        class_names = ["No-surgery", "surgery"]
    elif label_type == "preindex prostate cancer":
        class_names = ["no-cancer", "prostate-cancer"]
    elif label_type == "preindex cancer":
        class_names = ["no-cancer", "cancer"]
    else:
        raise NotImplementedError("unsuported target!!")
    return class_names
