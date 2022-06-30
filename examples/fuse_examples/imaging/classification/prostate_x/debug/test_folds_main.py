import pickle
import pandas as pd
import numpy as np
from fuseimg.datasets import prostate_x
import os
def main():

    df_tal_folds = read_and_test_tal_folds()
    test_tals_data(df_tal_folds)
    data_dir = os.environ["PROSTATEX_DATA_PATH"]
    user_home_dir = os.environ["USER_HOME_PATH"]
    cache_dir = os.path.join(f"{user_home_dir}", 'fuse_examples', 'prostate_x', 'cache_dir_v2')
    label_type = prostate_x.ProstateXLabelType.ClinSig
    dataset = prostate_x.ProstateX.dataset(label_type=label_type, train=False,
                                           cache_dir=cache_dir, data_dir=data_dir)

    df_folds1 = df_read_and_test_folds('/projects/msieve_dev3/usr/ozery/fuse_examples/prostate_x/prostatex_8folds_v2.pkl')

def test_tals_data(df):
    n_patient_fold_pairs = (df['Patient ID']+ '_'+ df['fold'].astype(str)).nunique()
    print(df.groupby(['ggg', 'is_ClinSig'])['fid'].count())
    print(df.groupby(['ClinSig', 'is_ClinSig'])['fid'].count())
    print(df.groupby(['Patient ID'])['ClinSig'].mean().unique())
    dd = df.groupby(['Patient ID'])[['fid', 'is_ClinSig']].agg({'fid': 'count', 'is_ClinSig': 'mean'})

    dd.columns = ['#fids', '#patients']
    print(dd.groupby('#fids').count())

    dd.columns = ['#fids', 'E(label)']
    print(dd.groupby('#fids').mean())
    print("ok")
def df_read_and_test_folds(filename):
    with open(filename, 'rb') as infile:
        folds_dict = pickle.load(infile)

    folds_list = []
    index_list = []

    for fold, indexes in  folds_dict.items():
        index_list += indexes
        folds_list += [fold] * len(indexes)
    data = np.asarray([index_list, folds_list]).T
    folds_df = pd.DataFrame(data, columns=['index', 'fold'])
    return folds_df
def read_and_test_tal_folds():
    filename = '/projects/msieve_dev3/usr/common/prostatex_processed_files/dataset_prostate_x_folds_ver29062021_seed1.pickle'
    with open(filename, 'rb') as infile:
        folds_df_map = pickle.load(infile)

    df_fold_list = []
    n_patients = 0
    for fold in range(len(folds_df_map)):
        fold_name = f'data_fold{fold}'
        df_fold = folds_df_map[fold_name]
        df_fold['fold'] = fold
        df_fold_list.append((df_fold))
        n_patients_fold=df_fold['Patient ID'].nunique()
        n_patients += n_patients_fold
        print(f"fold {fold}: size={df_fold.shape[0]} #patients={n_patients_fold}")
    print(f"total number of (patient, fold) unique pairs={n_patients}")
    df_all = pd.concat(df_fold_list, axis=0)
    n_patients = df_all['Patient ID'].nunique()
    print("# patients=",n_patients)
    n_patient_fold_pairs = (df_all['Patient ID']+ '_'+ df_all['fold'].astype(str)).nunique()
    print("# (patients, fold) pairs =", n_patient_fold_pairs)
    print("ok")
    return df_all

if __name__ == '__main__':
    main()