# import nibabel as nib

from fuse.utils.file_io.file_io import load_pickle, save_pickle_safe
from fuseimg.datasets.duke import Duke, get_selected_series_index
from fuse.data.utils.sample import create_initial_sample

import time
import numpy as np
from tqdm import tqdm
import os

def main():

    root_path = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'


    timestr = time.strftime("%Y%m%d-%H%M%S")

    if False:
        output_file = f'/user/ozery/output/duke_{timestr}.pkl'
        cache_dir = '/tmp/duke_cache_ps4y2jk'  # mkdtemp(prefix="duke_cache")
        duke_dataset = Duke.dataset(cache_dir=cache_dir, num_workers=0)
        print("finished creating dataset")
        arr = [d.flatten() for d in duke_dataset]
        print(len(arr))
        print(arr[0].keys())
        if output_file is not None:
            save_pickle(arr, output_file)
            print("wrote", output_file)
    else:
        sample_id = 'Breast_MRI_596'  # 'Breast_MRI_120'#'Breast_MRI_900' #'Breast_MRI_127'
        sample_ids = Duke.sample_ids()
        for sample_id in tqdm(sample_ids):
            output_file = f'/user/ozery/output/{sample_id}_v0.pkl'
            if os.path.exists(output_file):
                continue
            static_pipeline = Duke.static_pipeline(root_path=root_path, select_series_func=get_selected_series_index)

            sample_dict = create_initial_sample(sample_id)
            sample_dict = static_pipeline(sample_dict)
            # print("ok", sample_dict.flatten().keys())
            # save_pickle_safe(sample_dict, output_file)
            # print("saved", output_file)

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



if __name__ == "__main__":
    baseline_output_file = '/user/ozery/output/baseline1.pkl'  # '/tmp/f2.pkl'
    output_file = '/user/ozery/output/f5.pkl'
    main()


    # compare_sample_dicts('/user/ozery/output/Breast_MRI_900_v0.pkl','/user/ozery/output/Breast_MRI_900_20220531-232611.pkl')
