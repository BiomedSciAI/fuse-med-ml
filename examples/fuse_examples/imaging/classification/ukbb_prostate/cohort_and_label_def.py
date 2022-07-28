
from typing import Optional

import numpy as np
import pandas as pd

from fuse.utils import NDict


def get_samples_for_cohort(cohort_config: NDict, clinical_data_file:str, seed:Optional[int]=222):
    df = pd.read_csv(clinical_data_file)
    sample_ids = df['file'].values
    selected = np.zeros(df.shape[0], dtype=bool)
    group_ids = cohort_config['group_ids']
    max_group_size = cohort_config[ 'max_group_size']
    max_group_size = None if max_group_size <= 0 else max_group_size
    np.random.seed(seed)
    for group_id in group_ids:
        if group_id == 'all':
            group_filter = np.ones(df.shape[0], dtype=bool)
        elif group_id == 'men':
            group_filter = df['is female']==0
        elif group_id == 'men_no_cancer':
            group_filter = (df['is female'] == 0) & (df['preindex cancer'] == 0)
        elif group_id == 'men_prostate_cancer':
            group_filter = (df['is female'] == 0) & (df['preindex prostate cancer'] == 1)
        elif group_id == 'men_prostate_cancer_no_prostatectomy':
            group_filter = (df['is female'] == 0) & (df['preindex prostate cancer'] == 1) & (df['preindex prostatectomy'] == 0)
        elif group_id == 'men_prostatectomy':
            group_filter = (df['is female'] == 0) & (df['preindex prostatectomy'] == 1)
        elif group_id == 'men_no_prostatectomy':
            group_filter = (df['is female'] == 0) & (df['preindex prostatectomy'] == 0)
        else:
            raise NotImplementedError(group_id)

        group_size = group_filter.sum()

        if max_group_size is not None and group_size > max_group_size:
            all_indexes = np.where(group_filter)[0]
            rand_perm = np.random.permutation(group_size)
            n_remove = group_size -max_group_size
            indexes_to_remove = all_indexes[rand_perm[:n_remove]]
            assert np.all(group_filter[indexes_to_remove])
            group_filter[indexes_to_remove] = False
            assert np.sum(group_filter) == max_group_size
            print( group_id, "size:", group_size, "=>", max_group_size, "First removed index=", indexes_to_remove[0])
        else:
            print(group_id, "size:", group_size)
        selected |= group_filter
    print("cohort size=", np.sum(selected))
    return sample_ids[selected].tolist()


def get_class_names(label_type:str):
    if label_type == "classification":
        class_names = ["Male", "Female","Male-prostate-excision"]
    elif label_type == "is female":
        class_names = ["Male", "Female"]
    elif label_type == "preindex prostatectomy":
        class_names = ["No-surgery", "surgery"]
    elif label_type == "preindex prostate cancer":
        class_names = ["no-cancer", "prostate-cancer"]
    else:
        raise NotImplementedError("unsuported target!!")
    return class_names
