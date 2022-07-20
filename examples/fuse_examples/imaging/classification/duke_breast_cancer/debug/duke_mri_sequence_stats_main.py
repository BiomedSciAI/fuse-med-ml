import fuse_examples.imaging.classification.duke_breast_cancer.debug
import fuse_examples.imaging.classification.duke_breast_cancer.debug.duke_debug_main
from fuseimg.datasets import duke
from fuseimg.data.ops import ops_mri
from fuse.utils.file_io.file_io import load_pickle, save_pickle_safe

from tqdm import tqdm
import os
import pandas as pd


def main():
    root_path = "/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/"
    data_path = os.path.join(root_path, "Duke-Breast-Cancer-MRI")
    metadata_path = os.path.join(root_path, "metadata.csv")

    series_desc_2_sequence_map = duke.get_series_desc_2_sequence_mapping(metadata_path)
    sample_ids = duke.Duke.sample_ids()

    filename = "/user/ozery/output/duke_samples.pkl"
    if not os.path.exists(filename):
        samples_info = get_samples_info(sample_ids, data_path, series_desc_2_sequence_map)
        save_pickle_safe(samples_info, filename)
    else:
        samples_info = load_pickle(filename)

    df, cols_2_group = get_samples_stats(samples_info)
    df_g = get_group_statistics(df.reset_index(), cols_2_group)
    df_g.to_csv("/user/ozery/output/duke_stats.csv", index=False)

    annotations_df = fuse_examples.imaging.classification.duke_breast_cancer.debug.get_duke_annotations_from_tal_df()
    df2 = df.loc[annotations_df["Patient ID"]]
    df2_g = get_group_statistics(df2.reset_index(), cols_2_group)
    df2_g.to_csv("/user/ozery/output/duke_stats2.csv", index=False)


def get_samples_stats(samples_info):
    DCE_mix_ph1_4 = [f"DCE_mix_ph{i + 1}" for i in range(4)]
    seq_ids = DCE_mix_ph1_4 + ["DCE_mix_ph", "UNKNOWN"]
    n_seq_ids = len(seq_ids)
    seq_id_pos_map = dict(zip(seq_ids, range(n_seq_ids)))

    rows = []
    for sample_id, sample_info in samples_info.items():
        sample_encoding = ["0"] * n_seq_ids
        for seq_id, seq_arr in sample_info.items():
            pos = seq_id_pos_map[seq_id]
            sample_encoding[pos] = str(len(seq_arr))
        rows.append([sample_id] + sample_encoding)

    df = pd.DataFrame(rows, columns=["sample_id"] + seq_ids)
    df["DCE_mix_ph1_4"] = "C" + df["DCE_mix_ph1"] + df["DCE_mix_ph2"] + df["DCE_mix_ph3"] + df["DCE_mix_ph4"]
    df = df.drop(DCE_mix_ph1_4, axis=1)
    cols_2_group = ["DCE_mix_ph1_4", "DCE_mix_ph", "UNKNOWN"]
    df = df[cols_2_group + ["sample_id"]]

    return df.set_index("sample_id"), cols_2_group


def get_group_statistics(df, cols_2_group):
    df2 = df.groupby(cols_2_group, as_index=False).count()
    df2 = df2.rename({"sample_id": "# patients"}, axis=1)
    df2 = df2.sort_values("# patients")
    return df2


def get_samples_info(sample_ids, data_path, series_desc_2_sequence_map):
    samples_info = {}
    for sample_id in tqdm(sample_ids):
        sample_path = duke.get_sample_path(data_path, sample_id)
        seq_2_info_map = ops_mri.extract_seq_2_info_map(sample_path, series_desc_2_sequence_map)
        samples_info[sample_id] = seq_2_info_map
    return samples_info


if __name__ == "__main__":
    main()
