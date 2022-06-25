import os
os.environ["PROSTATEX_DATA_PATH"] = "/projects/msieve/MedicalSieve/PatientData/ProstateX/manifest-A3Y4AE4o5818678569166032044/"

import SimpleITK as sitk
from fuse.utils.file_io.file_io import load_pickle, save_pickle_safe
from fuseimg.datasets import prostate_x
from fuse.data.utils.sample import create_initial_sample
from deepdiff import DeepDiff
import numpy as np


import pandas as pd

def main():
    label_type = prostate_x.ProstateXLabelType.ClinSig
    # sample_ids = prostate_x.get_samples_for_debug(n_pos=10, n_neg=10,
    #                                         label_type=prostate_x.DukeLabelType.STAGING_TUMOR_SIZE)
    sample_ids = prostate_x.ProstateX.sample_ids()
    sample_ids = ['ProstateX-0010_1'] #
    sample_ids = ['ProstateX-0199_1'] #B-fix
    sample_ids = ['ProstateX-0030_1']
    sample_ids = ['ProstateX-0008_1']

    if True:
        static_pipeline = prostate_x.ProstateX.static_pipeline(root_path=os.environ["PROSTATEX_DATA_PATH"],
                                                               select_series_func=prostate_x.get_selected_series_index)

        if False:
            n_steps = 12 # 10 ok #9 is ok #7 is ok # 5 id ok
            static_pipeline._op_ids = static_pipeline._op_ids[:n_steps]
            static_pipeline._ops_and_kwargs = static_pipeline._ops_and_kwargs[:n_steps]
        dynamic_pipeline =  prostate_x.ProstateX.dynamic_pipeline(label_type=prostate_x.ProstateXLabelType.ClinSig,
                                                                  train=True)

        print("# sample_ids=", len(sample_ids))
        for sample_id in sample_ids:
            print(sample_id)
            sample_dict = create_initial_sample(sample_id)
            sample_dict = static_pipeline(sample_dict)
            sample_dict = dynamic_pipeline(sample_dict)
            print(sample_dict.flatten().keys())

            if False:
                # x_old = load_pickle('/tmp/ozery/t3.pkl') # n_steps == 9
                x_old = load_pickle('/tmp/ozery/t5.pkl')
                x_new = sitk.GetArrayFromImage(sample_dict['data.input.volume4D'])
                if np.all(x_old == x_new):
                    print( "ok")
                else:
                    print( "not ok")
            if False: # step 7
                arr_old= load_pickle('/tmp/ozery/t2.pkl')
                arr_new = [sitk.GetArrayFromImage(a) for a in sample_dict['data.input.selected_volumes']]
                assert len(arr_old) != len(arr_new)
                for i in range(len(arr_old)):
                    if np.all(arr_old[i] == arr_new[i]):
                        print(i, "ok")
                    else:
                        print(i, "not ok")

            print("oo")
            if False: # step 5
                d_old = load_pickle('/tmp/ozery/t1.pkl')

                d = {s: sample_dict[f'data.input.sequence.{s}'] for s in sample_dict['data.input.seq_ids'] }
                for k, v in d.items():
                    arr_new = [sitk.GetArrayFromImage(a['stk_volume']) for a in v]
                    arr_old = d_old[k]
                    if len(arr_new) != len(arr_old):
                        print(f"{k} mismatch in length")
                    else:
                        for ii in range(len(arr_new)):
                            if np.all(arr_new[ii] == arr_old[ii]):
                                print(f"{k} {ii} ok")
                            else:
                                print(f"{k} {ii} mismatch in content")

            print("ok")

        print("done")
    if False:
        # sample_ids = ['ProstateX-0058_1']#['ProstateX-0008_1']
        prostatex_dataset = prostate_x.ProstateX.dataset(data_dir=os.environ["PROSTATEX_DATA_PATH"],
                                                    label_type=prostate_x.ProstateXLabelType.ClinSig,
                                         cache_dir=None, num_workers=0, sample_ids=sample_ids)
        print("finished defining dataset, starting run")
        arr = []
        rows = []
        for d in prostatex_dataset:
            d2 = d.flatten()
            row = (d['data.sample_id'], d['data.ground_truth'])
            rows.append(row)
            print("*******", row)
            arr += [d2]
        print(len(arr))
        print(pd.DataFrame(rows, columns=['sample_id', 'gt']))
        print(arr[0].keys())


def compare_to_fuse1():
    fuse1_dir = '/tmp/ozery/prostatex_fuse1'
    prostatex_dataset = prostate_x.ProstateX.dataset(data_dir=os.environ["PROSTATEX_DATA_PATH"],
                                                     label_type=prostate_x.ProstateXLabelType.ClinSig,
                                                     cache_dir=None, num_workers=16, sample_ids=None,
                                                     verbose=False)
    deep_diff_config = dict(ignore_nan_inequality=True) #, math_epsilon=0.0001)
    n_errors = 0
    for i, sample_dict in enumerate(prostatex_dataset):
        sample_id = sample_dict['data.sample_id']
        fields = sample_id.split('_')
        filename = os.path.join(fuse1_dir, f'{fields[0]}_{int(fields[1])-1}.pkl')
        if not os.path.exists(filename):
            print(i, f"{filename} does not exist. skipping")
            n_errors += 1
            continue
        d_old =  load_pickle(filename)
        d_new = {'data.input':sample_dict[ 'data.input.patch_volume'].numpy(),
                                        'data.ground_truth':sample_dict[ 'data.ground_truth']}
        # diff = DeepDiff(d_old, d_new, **deep_diff_config)
        s_error = ''
        if d_old['data.ground_truth'] != d_new['data.ground_truth'].numpy():
            s_error += f' mismatch in label {d_old["data.ground_truth"]} in old'
        if not np.all(d_old['data.input'] == d_new['data.input']):
            s_error += ' mismatch in tensor'
        if len(s_error)>0:
            print(i, f"{filename} does not match: {s_error}")
            n_errors += 1
        else:
            print(i, f"{filename} ok.")
    print(f"Done. Total number of mismatches={n_errors}")

if __name__ == '__main__':
    main()
    # compare_to_fuse1()