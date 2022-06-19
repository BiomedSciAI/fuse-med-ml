# import nibabel as nib
from examples.fuse_examples.imaging.classification import duke_breast_cancer
from fuse.utils.file_io.file_io import load_pickle, save_pickle_safe
from fuseimg.datasets import duke
from fuse.data.utils.sample import create_initial_sample

import time
import numpy as np
from tqdm import tqdm

import getpass
import os
os.environ["DUKE_DATA_PATH"] = "/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/"

import pandas as pd

def main():

    timestr = time.strftime("%Y%m%d-%H%M%S")

    if True:
        # sample_ids=['Breast_MRI_900']
        sample_ids =  duke.get_samples_for_debug(data_dir=os.environ["DUKE_DATA_PATH"], n_pos=10, n_neg=10,
                                                 label_type=duke.DukeLabelType.STAGING_TUMOR_SIZE)
        duke_dataset = duke.Duke.dataset(data_dir=os.environ["DUKE_DATA_PATH"], label_type=duke.DukeLabelType.STAGING_TUMOR_SIZE,
                                         cache_dir=None, num_workers=0, sample_ids=sample_ids)
        print("finished defining dataset, starting run")
        arr = []
        rows = []
        for d in duke_dataset:
            d2 = d.flatten()
            row = (d['data.sample_id'], d['data.ground_truth'])
            rows.append(row)
            print("*******", row)
            arr +=[d2]
        print(len(arr))
        print(pd.DataFrame(rows, columns=['sample_id', 'gt']))
        print(arr[0].keys())

    else:
        sample_id = 'Breast_MRI_596'  # 'Breast_MRI_120'#'Breast_MRI_900' #'Breast_MRI_127'
        sample_ids = duke.Duke.sample_ids()[:5]
        for sample_id in tqdm(sample_ids):
            output_file = f'/user/ozery/output/{sample_id}_v0.pkl'
            if os.path.exists(output_file):
                continue
            static_pipeline = duke.Duke.static_pipeline(data_dir=root_path, select_series_func=duke.get_selected_series_index)

            sample_dict = create_initial_sample(sample_id)
            sample_dict = static_pipeline(sample_dict)
            # print("ok", sample_dict.flatten().keys())
            # save_pickle_safe(sample_dict, output_file)
            # print("saved", output_file)

def derive_fuse2_folds_files():
    input_filename_pattern = os.path.join(duke.DUKE_PROCESSED_FILE_DIR, "dataset_DUKE_folds_ver{name}_seed1.pickle")
    output_path = f'/projects/msieve_dev3/usr/{getpass.getuser()}/fuse_examples/duke'
    output_filename_pattern = os.path.join(output_path, "DUKE_folds_fuse2_{name}_seed1.pkl")

    for name in ['10012022Recurrence', '11102021TumorSize']:
        input_filename = input_filename_pattern.format(name=name)
        output_filename = output_filename_pattern.format(name=name)
        dict_in = load_pickle(input_filename)

        dict_out = {fold: dict_in[f'data_fold{fold}']['Patient ID'].values.tolist() for fold in range(5)}
        a = [len(v) for v in dict_out.values()]
        print(sum(a), a)
        save_pickle_safe(dict_out, output_filename)
        print("wrote", output_filename)


def compare_sample_dicts(file1, file2):
    d1 = load_pickle(file1).flatten()
    d2 = load_pickle(file2).flatten()
    set1 = set(d1.keys())
    set2 = set(d2.keys())
    if set1==set2:
        print("same keys")
    else:
        print("set1-set2", set1-set2, "set2-set1", set2-set1)
    for k in set1.intersection(set2):
        v1 = d1[k]
        v2 = d2[k]
        print(k, type(v1), type(v1)== type(v2), type(v1[0]) if isinstance(v1, list) else '', type(v2[0]) if isinstance(v2, list) else '')
        if k in ['data.input.volume4D', 'data.input.ref_volume',
                 'data.input.volumes.DCE_mix', 'data.input.selected_volumes', 'data.input.selected_volumes_resampled']:
            continue
        if type(v1).__module__ == np.__name__:
            assert (v1==v2).all()
        elif isinstance(v1, list):
            assert ((np.asarray(v1)==np.asarray(v2)).all())
        else:
            assert (v1==v2)
        print("OK")

    # map = {'data.sample_id': 'data.sample_id', 'data.input.sequence_ids': 'data.input.sequence_ids',
    #  'data.input.sequence_path.DCE_mix': 'data.input.path.DCE_mix',
    #  'data.input.sequence_volumes.DCE_mix': 'data.input.volumes.DCE_mix',
    #  'data.input.sequence_selected_volume.DCE_mix': 'data.input.selected_volumes.DCE_mix',
    #  'data.input.sequence_selected_path.DCE_mix': 'data.input.selected_path.DCE_mix',
    #  'data.input.sequence_selected_volume_resampled.DCE_mix': 'data.input.selected_volumes_resampled.DCE_mix'}
    #
    # for s1, s2 in map.items():
    #     print(s1, s2, s2 in d2.keys(), d1[s1]==d2[s2])


def print_excluded_patients():
    df = duke.get_duke_annotations_df()
    all_sample_ids = duke.Duke.sample_ids()
    print(df.shape, df.columns[0], len(all_sample_ids))
    excluded_sample_ids = sorted(list(set(all_sample_ids) - set(df.iloc[:,0].values)))
    print("excluded:",len(excluded_sample_ids))
    print([s[11:] for s in excluded_sample_ids])

def compare_old_new_train_dataset():
    files = ['/tmp/michal_old.pkl', '/tmp/michal_new.pkl']
    d_list_arr = [load_pickle(f) for f in files]

    print_sample_ids = lambda d_list: [d['data.sample_id'] for d in d_list]

    print(print_sample_ids(d_list_arr[0]) == print_sample_ids(d_list_arr[1]))
    for i in range(2):
        print(files[i], d_list_arr[i][0].flatten().keys())

    def get_dataset_dict(d_list):
        a_dict = {}
        for d in d_list:
            sample_id = d['data.sample_id']
            a_dict[sample_id] = duke_breast_cancer.replace_tensors_with_numpy(d.flatten())
        return a_dict
    dict_arr = []
    for i in range(2):
        dict_arr.append(get_dataset_dict(d_list_arr[i]))
    duke_breast_cancer.compare_dicts(dict_arr[0], dict_arr[1])

    print("ok")


if __name__ == "__main__":
    baseline_output_file = '/user/ozery/output/baseline1.pkl'  # '/tmp/f2.pkl'
    output_file = '/user/ozery/output/f5.pkl'
    # main()
    # print_excluded_patients()
    compare_old_new_train_dataset()


    # compare_sample_dicts('/user/ozery/output/Breast_MRI_900_v0.pkl','/user/ozery/output/Breast_MRI_900_20220531-232611.pkl')
    # derive_fuse2_folds_files()
