
from typing import Optional

import numpy as np
import pandas as pd

from fuse.utils import NDict


def get_samples_for_cohort(cohort_config: NDict, clinical_data_file:str, seed:Optional[int]=222, lgr=None):

    def write_log_info(s):
        if lgr is not None:
            lgr.info(s)

    df = pd.read_csv(clinical_data_file)
    var_namespace = get_clinical_vars_namespace(df)

    filter_out_group = cohort_config.get("filter_out")
    if filter_out_group is not None:
        filter_out = eval(filter_out_group, var_namespace)
    else:
        filter_out = None

    max_group_size = cohort_config[ 'max_group_size']
    max_group_size = None if max_group_size <= 0 else max_group_size
    np.random.seed(seed)

    selected = np.zeros(df.shape[0], dtype=bool)
    included_groups = cohort_config['groups']

    for included_group in included_groups:
        group_filter = eval(included_group, var_namespace)
        group_size = group_filter.sum()

        write_log_info(f'{included_group} size={group_size}')
        if filter_out is not None:
            group_filter &= ~filter_out
            group_size = group_filter.sum()
            write_log_info(f'{included_group} size={group_size} after filtering')

        if max_group_size is not None and group_size > max_group_size:
            all_indexes = np.where(group_filter)[0]
            rand_perm = np.random.permutation(group_size)
            n_remove = group_size -max_group_size
            indexes_to_remove = all_indexes[rand_perm[:n_remove]]
            assert np.all(group_filter[indexes_to_remove])
            group_filter[indexes_to_remove] = False
            assert np.sum(group_filter) == max_group_size
            write_log_info( f"{included_group} size: {group_size} => {max_group_size}, First removed index= {indexes_to_remove[0]}")

        selected |= group_filter
    print("cohort size=", np.sum(selected))
    return df['file_pattern'].values[selected].tolist()



def get_clinical_vars_namespace(df):
    mapping = {col.replace(' ', '_'): df[col]>0 for i, col in enumerate(df.columns) if df.dtypes[i] != 'O'}

    # vars in use: 'preindex_neoplasms', 'postindex_neoplasms', 'preindex_cancer', 'postindex_cancer'
    # 'prostate_hyperplasia_preindex', 'prostate_hyperplasia_postindex',
    # 'preindex_prostatectomy', 'postindex_prostatectomy',
    # 'preindex_prostate_resection',  'postindex_prostate_resection'

    mapping['women'] = mapping['is_female']
    mapping['men'] = ~mapping['is_female']

    acronyms = [('preindex_cancer_prostate', 'preindex_C61_Malignant_neoplasm_of_prostate'),
                ('postindex_cancer_prostate', 'postindex_C61_Malignant_neoplasm_of_prostate'),
                ('preindex_cancer_male_genital', 'preindex_C60-C63_Malignant_neoplasms_of_male_genital_organs'),
                ('postindex_cancer_male_genital', 'postindex_C60-C63_Malignant_neoplasms_of_male_genital_organs'),
                ('preindex_cancer_urinary_tract', 'preindex_C64-C68_Malignant_neoplasms_of_urinary_tract'),
                ('postindex_cancer_urinary_tract', 'postindex_C64-C68_Malignant_neoplasms_of_urinary_tract'),
                ('preindex_kidney_cancer','preindex_C64_Malignant_neoplasm_of_kidney,_except_renal_pelvis'),
                ('postindex_kidney_cancer', 'postindex_C64_Malignant_neoplasm_of_kidney,_except_renal_pelvis'),
                ]
    for s_new, s_old in acronyms:
        if s_old in mapping:
            mapping[s_new] = mapping[s_old]
        else:
            print(f"*** {s_old} does not exist in the clinical data file")

    return mapping

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
        class_names = [f'no {label_type}', label_type]
    return class_names
