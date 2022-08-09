from hydra import compose, initialize
from fuse.utils import file_io
import numpy as np
import pandas as pd


def main():
    cfg_overrides = ['target=prepostindex_cancer_prostate', 'cohort=cohort_cancer_prostate_prepostindex', 'max_group_size=1000']

    initialize(config_path="conf", job_name="test_app")  # only allows relative path
    cfg = compose(config_name="config", overrides=cfg_overrides)

    input_split_file = cfg["paths"]["data_split_filename"]
    print("using split file", input_split_file)
    output_file = input_split_file.replace(".pkl", "_for_genetics.csv") #.csv

    folds = file_io.load_pickle(input_split_file)
    assert len(folds) == 5
    print("Using patients in folds 0, 1,2 as as train")
    df_list = []
    for fold in range(5):
        sample_ids = folds[fold]
        patient_ids = [s.split('_')[0] for s in sample_ids]
        patient_ids = np.asarray(list(set(patient_ids)))
        df = pd.DataFrame(patient_ids.reshape(-1, 1), columns=['eid'])
        df['is_test'] = 0 if fold <= 2 else 1
        df_list.append(df)

    df_all = pd.concat(df_list, axis=0)
    assert df_all.eid.nunique() == df_all.shape[0]
    df_all.to_csv(output_file, index=False)
    print("wrote", output_file)

    df_test = pd.read_csv(output_file)
    print(df_test.groupby('is_test').count())

if __name__ == '__main__':
    main()
