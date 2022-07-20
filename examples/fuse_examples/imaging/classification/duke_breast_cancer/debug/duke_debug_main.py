# import nibabel as nib

import fuseimg.datasets.duke_label_type
from fuse_examples.imaging.classification import duke_breast_cancer
from fuse_examples.imaging.classification.duke_breast_cancer.debug import (
    DUKE_PROCESSED_FILE_DIR,
    get_duke_annotations_from_tal_df,
    get_col_mapping,
)
from fuse.utils.file_io.file_io import load_pickle, save_pickle_safe
from fuseimg.datasets import duke
from fuse.data.utils.sample import create_initial_sample
from fuse.data.ops import ops_cast
import torch
from deepdiff import DeepDiff
import SimpleITK as sitk
from fuse.utils import file_io
import time
import numpy as np
from tqdm import tqdm

import getpass
import os

from fuseimg.datasets.duke import get_duke_clinical_data_df

import pandas as pd


def main():

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if False:
        sample_id = "Breast_MRI_900"

        static_pipeline = duke.Duke.static_pipeline(
            data_dir=os.environ["DUKE_DATA_PATH"],
            select_series_func=duke.get_selected_series_index,
        )
        # k = 5  #ok
        # k = 8
        # static_pipeline._ops_and_kwargs = static_pipeline._ops_and_kwargs[:k]
        # static_pipeline._op_ids = static_pipeline._op_ids[:k]

        sample_dict = create_initial_sample(sample_id)
        sample_dict = static_pipeline(sample_dict)
        s = replace_stk_with_numpy(sample_dict.flatten())
        s_old = file_io.load_pickle(f"/tmp/ozery/s_old_stat.pkl")
        # x_new = s['data.input.volume4D']
        # x_old = s_old['data.input.volume4D'][:,:,:,0]
        x_new = s["data.input.patch_volume"]
        x_old = s_old["data.input.patch_volume"]
        print("ok", np.abs(x_new - x_old).max())
        # deep_diff_config = dict(ignore_nan_inequality=True)  # , math_epsilon=0.0001)
        # diff = DeepDiff(s_old, s, **deep_diff_config)
        # if len(diff) > 0:
        #     print(diff.keys())

    if True:
        sample_ids = ["Breast_MRI_900"]

        # sample_ids =  duke.get_samples_for_debug(data_dir=os.environ["DUKE_DATA_PATH"], n_pos=10, n_neg=10,
        #                                          label_type=duke.DukeLabelType.STAGING_TUMOR_SIZE)
        duke_dataset = duke.Duke.dataset(
            data_dir=os.environ["DUKE_DATA_PATH"],
            label_type=fuseimg.datasets.duke_label_type.DukeLabelType.STAGING_TUMOR_SIZE,
            cache_dir=None,
            num_workers=0,
            sample_ids=sample_ids,
        )
        print("finished defining dataset, starting run")
        arr = []
        rows = []
        deep_diff_config = dict(ignore_nan_inequality=True)  # , math_epsilon=0.0001)
        for d in duke_dataset:
            d2 = replace_tensors_with_numpy(d.flatten())
            row = (d["data.sample_id"], d["data.ground_truth"])
            rows.append(row)
            print("*******", row)
            arr += [d2]
            output_file = f"/tmp/ozery/s{sample_ids[0]}.pkl"
            d2_old = replace_tensors_with_numpy(load_pickle(output_file).flatten())
            diff = DeepDiff(d2_old, d2, **deep_diff_config)
            if len(diff) > 0:
                print(sample_ids[0], "has diff")
            print("wrote", output_file)
            break
        print(len(arr))
        print(pd.DataFrame(rows, columns=["sample_id", "gt"]))
        print(arr[0].keys())

    else:
        sample_id = (
            "Breast_MRI_596"  # 'Breast_MRI_120'#'Breast_MRI_900' #'Breast_MRI_127'
        )
        sample_ids = duke.Duke.sample_ids()[:5]
        for sample_id in tqdm(sample_ids):
            output_file = f"/user/ozery/output/{sample_id}_v0.pkl"
            if os.path.exists(output_file):
                continue
            static_pipeline = duke.Duke.static_pipeline(
                data_dir=root_path, select_series_func=duke.get_selected_series_index
            )

            sample_dict = create_initial_sample(sample_id)
            sample_dict = static_pipeline(sample_dict)
            # print("ok", sample_dict.flatten().keys())
            # save_pickle_safe(sample_dict, output_file)
            # print("saved", output_file)


def replace_tensors_with_numpy(d):
    d2 = {}
    for k, v in d.items():

        if isinstance(v, torch.Tensor):
            v = ops_cast.Cast.to_numpy(v)
            print(k, "tensor => numpy")
        d2[k] = v

    return d2


def derive_fuse2_folds_files():
    input_filename_pattern = os.path.join(
        DUKE_PROCESSED_FILE_DIR, "dataset_DUKE_folds_ver{name}_seed1.pickle"
    )
    output_path = f"/projects/msieve_dev3/usr/{getpass.getuser()}/fuse_examples/duke"
    output_filename_pattern = os.path.join(
        output_path, "DUKE_folds_fuse2_{name}_seed1.pkl"
    )

    for name in ["10012022Recurrence", "11102021TumorSize"]:
        input_filename = input_filename_pattern.format(name=name)
        output_filename = output_filename_pattern.format(name=name)
        dict_in = load_pickle(input_filename)

        dict_out = {
            fold: dict_in[f"data_fold{fold}"]["Patient ID"].values.tolist()
            for fold in range(5)
        }
        a = [len(v) for v in dict_out.values()]
        print(sum(a), a)
        save_pickle_safe(dict_out, output_filename)
        print("wrote", output_filename)


def compare_sample_dicts(file1, file2):
    d1 = load_pickle(file1).flatten()
    d2 = load_pickle(file2).flatten()
    set1 = set(d1.keys())
    set2 = set(d2.keys())
    if set1 == set2:
        print("same keys")
    else:
        print("set1-set2", set1 - set2, "set2-set1", set2 - set1)
    for k in set1.intersection(set2):
        v1 = d1[k]
        v2 = d2[k]
        print(
            k,
            type(v1),
            type(v1) == type(v2),
            type(v1[0]) if isinstance(v1, list) else "",
            type(v2[0]) if isinstance(v2, list) else "",
        )
        if k in [
            "data.input.volume4D",
            "data.input.ref_volume",
            "data.input.volumes.DCE_mix",
            "data.input.selected_volumes",
            "data.input.selected_volumes_resampled",
        ]:
            continue
        if type(v1).__module__ == np.__name__:
            assert (v1 == v2).all()
        elif isinstance(v1, list):
            assert (np.asarray(v1) == np.asarray(v2)).all()
        else:
            assert v1 == v2
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


def get_excluded_patients(do_print=True):
    df = get_duke_annotations_from_tal_df()
    all_sample_ids = duke.Duke.sample_ids()
    print(df.shape, df.columns[0], len(all_sample_ids))
    excluded_sample_ids = sorted(list(set(all_sample_ids) - set(df.iloc[:, 0].values)))
    if do_print:
        print("excluded:", len(excluded_sample_ids))
        print([s[11:] for s in excluded_sample_ids])
    return excluded_sample_ids


def check_fuse_results():
    data_dir = duke_breast_cancer.get_duke_user_dir()
    filename = os.path.join(data_dir, "model_dir/infer_dir/validation_set_infer.gz")
    df = load_pickle(filename)
    print(df.shape, df.columns)
    excluded_sample_ids = get_excluded_patients(do_print=False)
    excluded_in_df = set(df.id).intersection(set(excluded_sample_ids))
    print("ok")


def visualize_image_from_cache():
    data_dir = duke_breast_cancer.get_duke_user_dir()
    data_dir2 = os.path.join(
        data_dir, "cache_dir/duke_cache_ver0/hash_b03e85135f7b2b2ad5aa02b372920317"
    )
    filename = os.path.join(
        data_dir2, "out_sample_id@ffebf1d99a8358183f7b031c842c7c84.pkl.gz"
    )
    sample_dict = load_pickle(filename)
    dynamic_pipeline = duke.Duke.dynamic_pipeline(
        include_patch_by_mask=True, include_patch_fixed=True, verbose=True
    )
    sample_dict = dynamic_pipeline(sample_dict)

    print(type(sample_dict))


def replace_tensors_with_numpy(d):
    d2 = {}
    for k, v in d.items():

        if isinstance(v, torch.Tensor):
            v = ops_cast.Cast.to_numpy(v)
            print(k, "tensor => numpy")
        d2[k] = v
    return d2


def replace_stk_with_numpy(d):
    d2 = {}
    for k, v in d.items():
        if isinstance(v, sitk.Image):
            v = sitk.GetArrayFromImage(v)
            print(k, "tensor => numpy")
        d2[k] = v
    return d2


def f(d):
    from fuse.utils import file_io

    file_io.save_pickle_safe(d, "/tmp/ozery/s_old.pkl")
    from fuse.utils import file_io

    s_old = file_io.load_pickle("/tmp/ozery/s_old.pkl")
    s = None

    from deepdiff import DeepDiff

    s_old = file_io.load_pickle("/tmp/ozery/s_old.pkl")
    deep_diff_config = dict(ignore_nan_inequality=True)  # , math_epsilon=0.0001)
    diff = DeepDiff(s_old, s, **deep_diff_config)
    if len(diff) > 0:
        print(diff.keys())


def cmp_features():
    data_dir = "/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI"

    map = get_col_mapping()
    cols_new = list(map.keys())
    cols_old = [map[k] for k in cols_new]
    df_new = get_duke_clinical_data_df(data_dir).set_index(
        "Patient Information:Patient ID"
    )[cols_new]
    df_old = get_duke_annotations_from_tal_df().set_index("Patient ID DICOM")[cols_old]
    df_new = df_new.loc[df_old.index]
    for icol in range(len(map)):
        v_new = df_new.iloc[:, icol]
        v_old = df_old.iloc[:, icol]

        v_new0 = v_new[0]
        v_old0 = v_old[0]
        # print("**", cols_old[icol], v_new0, v_old0, type(v_new0), type(v_old0))
        is_same = v_new == v_old
        if isinstance(v_new0, np.float64) and isinstance(v_old0, np.float64):
            is_same |= np.isnan(v_new) & np.isnan(v_old)
        if (is_same).all():
            print(cols_old[icol], "OK")
        else:
            ix = np.where(~is_same)[0]
            diff = ~is_same
            print(
                "----",
                cols_old[icol],
                "ERROR!!!",
                df_new.index[ix[0]],
                v_new[ix[0]],
                v_old[ix[0]],
                f"#diff={diff.sum()} / {diff.shape[0]}",
            )


if __name__ == "__main__":
    baseline_output_file = "/user/ozery/output/baseline1.pkl"  # '/tmp/f2.pkl'
    output_file = "/user/ozery/output/f5.pkl"
    main()
    # get_excluded_patients(do_print=True)

    # compare_sample_dicts('/user/ozery/output/Breast_MRI_900_v0.pkl','/user/ozery/output/Breast_MRI_900_20220531-232611.pkl')
    # derive_fuse2_folds_files()
    # check_fuse_results()
    # visualize_image_from_cache()
