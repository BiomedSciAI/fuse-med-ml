import os
import numpy as np
import pandas as pd
import pickle
from ast import literal_eval as make_tuple

# todo: remove setting of env variable !!!
os.environ["DUKE_DATA_PATH"] = "/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/"
from fuseimg.datasets import duke
from fuse.utils.utils_debug import FuseDebug


from fuse.utils.utils_logger import fuse_logger_start
from typing import OrderedDict
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve
from examples.fuse_examples.imaging.classification import duke_breast_cancer
def main():
    dir2 = '/projects/msieve_dev3/usr/common/duke_processed_files'
    files = ['dataset_DUKE_folds_ver10012022Recurrence_seed1.pickle',
             'dataset_DUKE_folds_ver11102021TumorSize_seed1.pickle']
    df_list = [read_df(os.path.join(dir2, f)) for f in files]

    selected_sample_ids = ['Breast_MRI_900']
    params = dict(cache_dir=os.path.join(duke_breast_cancer.get_duke_lesion_properties_user_dir(),'cache_dir'),
                  reset_cache=False,
                  sample_ids=selected_sample_ids, num_workers=0,
                  select_series_func=duke.get_selected_series_index,
                  cache_kwargs=dict(audit_first_sample=False, audit_rate=None)  # None
                  )
    dataset_all = duke.DukeLesionProperties.dataset(**params)
    rows = []
    for sample_dict in dataset_all:
        sample_dict2 = dict(sample_id= sample_dict['data.sample_id'])
        for k, v, in sample_dict['data.lesion_properties'].items():
            sample_dict2[k] = v
        rows.append(sample_dict2)
    df = pd.DataFrame(rows).set_index('sample_id')
    for df2 in df_list:
        indexes= set(df.index).intersection(df2.index)
        df2.columns = [s.replace('_T0', '') for s in df2.columns]
        missing_cols = set(df.columns) - set(df2.columns)
        compared_cols = list(set(df.columns).intersection(set(df2.columns)))
        print("missing columne", missing_cols)
        print("compared cold", compared_cols)

        for index in indexes:
            row = df.loc[index]
            row2 = df2.loc[index]
            for col in compared_cols:
                v = row[col]
                v2 = row2[col]
                if isinstance(v, tuple):
                    v2  = make_tuple(v2)
                    for i in range(len(v)):
                        assert np.abs(v[i]-v2[i]) < 0.0001
                else:
                    assert np.abs(v - v2) < 0.0001
                # if
                # assert row[col] == row2[col]
        print("---------ok---------")



def read_df(df_file):
    with open(df_file, 'rb') as infile:
        fold_annotations_dict = pickle.load(infile)

    return pd.concat(fold_annotations_dict.values(), axis=0).set_index('Patient ID DICOM')


if __name__ == '__main__':
    main()