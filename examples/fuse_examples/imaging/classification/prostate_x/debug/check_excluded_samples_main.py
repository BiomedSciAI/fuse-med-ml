import numpy as np
import pandas as pd
import os
from fuseimg.datasets import prostate_x


def main():
    data_dir = os.environ["PROSTATEX_DATA_PATH"]
    all_anotations_df = get_all_annotations(data_dir)
    cache_dir = ""

    label_type = prostate_x.ProstateXLabelType.ClinSig
    # bad examples:
    # 'ProstateX-0025' - entirely excluded
    # 'ProstateX-0005' - 2,3  [1 is used]
    # 'ProstateX-0105' - 2,3 [1 is used]
    # 'ProstateX-0154' - 3 [1,2, are used]
    sample_ids = ["ProstateX-0005_2"]

    dataset = prostate_x.ProstateX.dataset(
        label_type=label_type,
        train=False,
        cache_dir=cache_dir,
        data_dir=data_dir,
        sample_ids=sample_ids,
        annotations_df=all_anotations_df,
    )


def get_all_annotations(data_dir):
    annotations_df = pd.read_csv(os.path.join(data_dir, "Lesion Information", "ProstateX-Findings-Train.csv"))
    annotations_df["Patient ID"] = annotations_df["ProxID"]
    annotations_df = annotations_df.set_index("ProxID")
    annotations_df = annotations_df[["Patient ID", "fid", "ClinSig", "pos", "zone"]]

    pids_to_fix = [("ProstateX-0159", 3), ("ProstateX-0005", 3), ("ProstateX-0025", 5)]
    for pid, n_fid in pids_to_fix:
        a_filter = annotations_df["Patient ID"] == pid
        assert a_filter.sum() == n_fid
        annotations_df.loc[a_filter, "fid"] = np.arange(1, n_fid + 1)
    annotations_df["Sample ID"] = annotations_df["Patient ID"] + "_" + annotations_df["fid"].astype(str)
    a_filter = annotations_df["Sample ID"].duplicated()
    print(annotations_df[a_filter]["Sample ID"])


if __name__ == "__main__":
    main()
