import typing
from typing import Optional

import numpy as np
import pandas as pd

from fuse.utils import NDict


def get_samples_for_cohort(cohort_config: NDict, var_namespace:typing.Dict, seed:Optional[int]=222, lgr=None):

    def write_log_info(s):
        if lgr is not None:
            lgr.info(s)

    max_group_size = cohort_config[ 'max_group_size']
    max_group_size = None if max_group_size <= 0 else max_group_size
    np.random.seed(seed)

    selected = eval(cohort_config['inclusion'], var_namespace)

    y = var_namespace[cohort_config['group_id_vec']]
    y_vals = np.unique(y)

    n = 0

    for y_val in y_vals:
        group_filter = (y == y_val) & selected
        group_size = group_filter.sum()

        write_log_info(f'target={y_val} size={group_size}')

        if max_group_size is not None and group_size > max_group_size:
            all_indexes = np.where(group_filter)[0]
            rand_perm = np.random.permutation(group_size)
            n_remove = group_size -max_group_size
            indexes_to_remove = all_indexes[rand_perm[:n_remove]]
            assert np.all(group_filter[indexes_to_remove])
            selected[indexes_to_remove] = False
            group_filter[indexes_to_remove] = False
            assert np.sum(group_filter) == max_group_size
            write_log_info( f"target={y_val} size: {group_size} => {max_group_size}, First removed index= {indexes_to_remove[0]}")
        n += np.sum(group_filter)
    print("cohort size=", np.sum(selected))
    assert np.sum(selected) == n
    return var_namespace['file_pattern'][selected].tolist()



def get_clinical_vars_namespace(df, columns_to_add):
    var_namespace = {col.replace(' ', '_').replace(',', '_').replace('-', '_'):
                         df[col] for i, col in enumerate(df.columns) }

    for col_name, col_expression in columns_to_add.items():
        x = eval(col_expression, var_namespace)
        var_namespace[col_name] = x
    return var_namespace

def get_class_names(label_type:str):
    #todo: need to revisit. define class names in config
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
        class_names = [f'no {label_type}', label_type]
    return class_names
