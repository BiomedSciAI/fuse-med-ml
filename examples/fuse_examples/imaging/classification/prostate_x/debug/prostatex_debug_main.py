import os

import SimpleITK as sitk
from fuse.utils.file_io.file_io import load_pickle, save_pickle_safe
from fuseimg.datasets import prostate_x
from fuse.data.utils.sample import create_initial_sample
from deepdiff import DeepDiff
import numpy as np
import pickle


import pandas as pd

PROSTATEX_PROCESSED_FILE_DIR = (
    "/projects/msieve_dev3/usr/Tal/prostate_x_processed_files"
)


def main():
    label_type = prostate_x.ProstateXLabelType.ClinSig
    # sample_ids = prostate_x.get_samples_for_debug(n_pos=10, n_neg=10,
    #                                         label_type=prostate_x.DukeLabelType.STAGING_TUMOR_SIZE)
    sample_ids = prostate_x.ProstateX.sample_ids(
        data_dir=os.environ["PROSTATEX_DATA_PATH"]
    )
    sample_ids = ["ProstateX-0010_1"]  #
    sample_ids = ["ProstateX-0199_1"]  # B-fix
    sample_ids = ["ProstateX-0030_1"]
    sample_ids = ["ProstateX-0159_1", "ProstateX-0159_2", "ProstateX-0159_3"]

    if True:
        static_pipeline = prostate_x.ProstateX.static_pipeline(
            root_path=os.environ["PROSTATEX_DATA_PATH"],
            select_series_func=prostate_x.get_selected_series_index,
        )

        if False:
            n_steps = 12  # 10 ok #9 is ok #7 is ok # 5 id ok
            static_pipeline._op_ids = static_pipeline._op_ids[:n_steps]
            static_pipeline._ops_and_kwargs = static_pipeline._ops_and_kwargs[:n_steps]
        dynamic_pipeline = prostate_x.ProstateX.dynamic_pipeline(
            label_type=prostate_x.ProstateXLabelType.ClinSig, train=True
        )

        print("# sample_ids=", len(sample_ids))
        for sample_id in sample_ids:
            print(sample_id)
            sample_dict = create_initial_sample(sample_id)
            sample_dict = static_pipeline(sample_dict)
            sample_dict = dynamic_pipeline(sample_dict)
            print(sample_dict.flatten().keys())

            if False:
                # x_old = load_pickle('/tmp/ozery/t3.pkl') # n_steps == 9
                x_old = load_pickle("/tmp/ozery/t5.pkl")
                x_new = sitk.GetArrayFromImage(sample_dict["data.input.volume4D"])
                if np.all(x_old == x_new):
                    print("ok")
                else:
                    print("not ok")
            if False:  # step 7
                arr_old = load_pickle("/tmp/ozery/t2.pkl")
                arr_new = [
                    sitk.GetArrayFromImage(a)
                    for a in sample_dict["data.input.selected_volumes"]
                ]
                assert len(arr_old) != len(arr_new)
                for i in range(len(arr_old)):
                    if np.all(arr_old[i] == arr_new[i]):
                        print(i, "ok")
                    else:
                        print(i, "not ok")

            print("oo")
            if False:  # step 5
                d_old = load_pickle("/tmp/ozery/t1.pkl")

                d = {
                    s: sample_dict[f"data.input.sequence.{s}"]
                    for s in sample_dict["data.input.seq_ids"]
                }
                for k, v in d.items():
                    arr_new = [sitk.GetArrayFromImage(a["stk_volume"]) for a in v]
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
        prostatex_dataset = prostate_x.ProstateX.dataset(
            data_dir=os.environ["PROSTATEX_DATA_PATH"],
            label_type=prostate_x.ProstateXLabelType.ClinSig,
            cache_dir=None,
            num_workers=0,
            sample_ids=sample_ids,
        )
        print("finished defining dataset, starting run")
        arr = []
        rows = []
        for d in prostatex_dataset:
            d2 = d.flatten()
            row = (d["data.sample_id"], d["data.ground_truth"])
            rows.append(row)
            print("*******", row)
            arr += [d2]
        print(len(arr))
        print(pd.DataFrame(rows, columns=["sample_id", "gt"]))
        print(arr[0].keys())


def compare_to_fuse1():
    fuse1_dir = "/tmp/ozery/prostatex_fuse1"
    prostatex_dataset = prostate_x.ProstateX.dataset(
        data_dir=os.environ["PROSTATEX_DATA_PATH"],
        label_type=prostate_x.ProstateXLabelType.ClinSig,
        cache_dir=None,
        num_workers=16,
        sample_ids=None,
        verbose=False,
    )
    deep_diff_config = dict(ignore_nan_inequality=True)  # , math_epsilon=0.0001)
    n_errors = 0
    for i, sample_dict in enumerate(prostatex_dataset):
        sample_id = sample_dict["data.sample_id"]
        fields = sample_id.split("_")
        filename = os.path.join(fuse1_dir, f"{fields[0]}_{int(fields[1])-1}.pkl")
        if not os.path.exists(filename):
            print(i, f"{filename} does not exist. skipping")
            n_errors += 1
            continue
        d_old = load_pickle(filename)
        d_new = {
            "data.input": sample_dict["data.input.patch_volume"].numpy(),
            "data.ground_truth": sample_dict["data.ground_truth"],
        }
        # diff = DeepDiff(d_old, d_new, **deep_diff_config)
        s_error = ""
        if d_old["data.ground_truth"] != d_new["data.ground_truth"].numpy():
            s_error += f' mismatch in label {d_old["data.ground_truth"]} in old'
        if not np.all(d_old["data.input"] == d_new["data.input"]):
            s_error += " mismatch in tensor"
        if len(s_error) > 0:
            print(i, f"{filename} does not match: {s_error}")
            n_errors += 1
        else:
            print(i, f"{filename} ok.")
    print(f"Done. Total number of mismatches={n_errors}")


def test_files():
    df = get_tal_prostatex_annotations_file()
    print(df.shape, df["Patient ID"].nunique(), df.columns.values)
    pids = set(df["Patient ID"])
    df["sample_id"] = df["Patient ID"] + "_" + df["fid"].astype(str)
    print(df.shape[0], df["sample_id"].nunique())

    filenames = ["ProstateX-2-Findings-Train.csv", "ProstateX-Findings-Train.csv"]
    filenames = [
        os.path.join(os.environ["PROSTATEX_DATA_PATH"], "Lesion Information", f)
        for f in filenames
    ]
    df_list = []
    for filename in filenames:
        assert os.path.exists(filename)
        df2 = pd.read_csv(filename).rename({"ProxID": "Patient ID"}, axis=1)

        a_filter = df2["Patient ID"] == "ProstateX-0159"
        if np.any(a_filter):
            assert a_filter.sum() == 3
            df2.loc[a_filter, "fid"] = [1, 2, 3]
        df2["sample_id"] = df2["Patient ID"] + "_" + df2["fid"].astype(str)

        df_list.append(df2)
        print(df2.shape, df2.columns.values)
        print(df2.iloc[0])
        print(df2.zone.unique(), df2["Patient ID"].nunique())
    df2 = df_list[1]
    df3 = df_list[0]

    if False:
        dd = df3.merge(df2, on=["sample_id", "fid", "Patient ID"])
        a_filter = (dd.pos_x != dd.pos_y) | (dd.zone_x != dd.zone_y)
        print(dd[a_filter])
        # change in one record: ProstateX-0005
        # values of  'ProstateX-Findings-Train.csv' matches Tal's file

    # compare df2 and df:
    dd = df.merge(df2, on=["sample_id", "fid", "Patient ID"])
    cols_2_compare = [s for s in dd.columns if s.endswith("_x")]
    print("comparing to between Tal's processd file and", filenames[1])
    for sx in cols_2_compare:
        sy = sx[:-1] + "y"
        a_filter = dd[sx] != dd[sy]
        print("checking", sx)
        if np.any(a_filter):
            print(dd[["sample_id", sx, sy]].loc[0])

    print(sorted(list(set(df2["sample_id"]) - set(df["sample_id"]))))
    print("ok")


def get_tal_prostatex_annotations_file():
    annotations_path = os.path.join(
        PROSTATEX_PROCESSED_FILE_DIR,
        "dataset_prostate_x_folds_ver29062021_seed1.pickle",
    )
    with open(annotations_path, "rb") as infile:
        fold_annotations_dict = pickle.load(infile)
    annotations_df = pd.concat(
        [
            fold_annotations_dict[f"data_fold{fold}"]
            for fold in range(len(fold_annotations_dict))
        ]
    )

    for pid, n_fid in [("ProstateX-0025", 5), ("ProstateX-0159", 3)]:
        a_filter = annotations_df["Patient ID"] == pid
        if np.any(a_filter):
            assert a_filter.sum() == n_fid
            annotations_df.loc[a_filter, "fid"] = np.arange(1, n_fid + 1)
    annotations_df["Sample ID"] = annotations_df[["Patient ID", "fid"]].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )

    return annotations_df


def compare_datasets_on_dict():
    dir1 = "/tmp/ozery/prostatex_v5"  # bad
    dir2 = "/tmp/ozery/prostatex_v6"  # goood
    deep_diff_config = dict(ignore_nan_inequality=True)  # , math_epsilon=0.0001)
    for i, file in enumerate(sorted(os.listdir(dir2))):
        df_list = []
        for dirx in [dir1, dir2]:
            filex = os.path.join(dirx, file)
            # if not os.path.exists(filex):
            #     print(file, "does not exist")
            #     continue
            d = load_pickle(filex)
            df_list.append(d)
        diff = DeepDiff(df_list[0], df_list[1], **deep_diff_config)
        if len(diff) > 0:
            print(i, file, diff.keys())
        else:
            print(
                i,
                "ok",
                df_list[0]["data.ground_truth"],
                df_list[1]["data.ground_truth"],
            )
    print("done")


if __name__ == "__main__":
    # main()
    # compare_to_fuse1()
    test_files()
    # compare_datasets_on_dict()
