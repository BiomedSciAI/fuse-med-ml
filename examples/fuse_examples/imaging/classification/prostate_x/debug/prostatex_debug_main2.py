import os
os.environ["PROSTATEX_DATA_PATH"] = "/projects/msieve/MedicalSieve/PatientData/ProstateX/manifest-A3Y4AE4o5818678569166032044/"

from fuse.utils.file_io.file_io import load_pickle, save_pickle_safe
from fuseimg.datasets import prostate_x
from fuse.data.utils.sample import create_initial_sample

import pandas as pd

def main():
    label_type = prostate_x.ProstateXLabelType.ClinSig
    # sample_ids = prostate_x.get_samples_for_debug(n_pos=10, n_neg=10,
    #                                         label_type=prostate_x.DukeLabelType.STAGING_TUMOR_SIZE)
    sample_ids = prostate_x.ProstateX.sample_ids()[:4]
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


if __name__ == '__main__':
    main()