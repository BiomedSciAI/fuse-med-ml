"""
(C) Copyright 2021 IBM Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Created on June 30, 2021
"""

import os
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from fuse.data.datasets.dataset_base import DatasetBase
from fuse.data.utils.export import ExportDataset
from fuse.data.utils.sample import get_sample_id_key
from fuse.utils.file_io.file_io import load_pickle, save_pickle


def print_folds_stat(db: pd.DataFrame, nfolds: int, key_columns: np.ndarray) -> str:
    """
    Print fold statistics
    :param db:                 dataframe which contains the fold partition
    :param nfolds:             Number of folds to divide the data
    :param key_columns:        keys for which balancing is forced
    """
    result = ""
    for f in range(nfolds):
        for key in key_columns:
            result += "----------fold" + str(f) + "\n"
            result += "key: " + key + "\n"
            result += db[db["fold"] == f][key].value_counts().to_string() + "\n"
    return result


def balanced_division(
    df: pd.DataFrame,
    no_mixture_id: str,
    keys_to_balance: Sequence[str],
    nfolds: int,
    seed: int = 1357,
    excluded_samples: Sequence[Hashable] = tuple(),
    print_flag: bool = False,
    debug_mode: bool = False,
) -> pd.DataFrame:
    """
    Partition the data into folds while using no_mixture_id for which no mixture between folds should be forced.
    and using keys_to_balance as the keys for which balancing is forced.
    The functions creates ID level labeling which is the cross-section of all possible mixture of key columns for that id
    it creates the folds so each fold will have about same proportions of ID level labeling while each ID will appear only in one fold
    For example - patient with ID 1234 has 2 images , each image has a binary classification (benign / malignant) .
    it can be that both of his images are benign or both are malignant or one is benign and the other is malignant.
    :param df:                 dataframe containing all samples including id and keys_to_balance
    :param no_mixture_id:      The key column for which no mixture between folds should be forced
    :param keys_to_balance:        keys for which balancing is forced
    :param nfolds:              number of folds to divide the data
    :param seed:               random seed used for creating folds
    :param excluded_samples:   sampled id which we do not want to include in the folds
    :param print_flag:         boolean flag which indicates if to print fold statistics
    """
    id_level_labels = []
    record_labels = []
    for field in keys_to_balance:
        values = df[field].unique()
        for value in values:
            value2 = str.replace(str(value), "+", "")
            # creates a binary label for each record and label
            record_key = "is" + value2
            df[record_key] = df[field] == value
            # creates a binary label for each id and label ( is anyone with this id has his label)
            id_level_key = "sample_id_" + field + "_" + value2
            df[id_level_key] = (
                df.groupby([no_mixture_id])[record_key].transform(sum) > 0
            )
            id_level_labels.append(id_level_key)
            record_labels.append(record_key)

    # drop duplicate id records
    samples_col = [no_mixture_id] + list(id_level_labels)
    df_samples = df[samples_col].drop_duplicates()

    # generates a new label for each id based on sample_id value, using id's which are not in excluded_samples
    excluded_samples_df = df_samples[no_mixture_id].isin(excluded_samples)
    included_samples_df = df_samples[id_level_labels][~excluded_samples_df]
    df_samples["y_class"] = [str(t) for t in included_samples_df.values]
    y_values = list(df_samples["y_class"].unique())

    # initialize folds to empty list of ids
    db_samples = {}
    for f in range(nfolds):
        db_samples["data_fold" + str(f)] = []

    # creates a dictionary with key=fold , and values = ID which is in the fold
    # the partition goes as following : for each id level labels we shuffle the ID's and split equally ( as possible) to nfolds
    for y_value in y_values:
        patients_w_value = list(
            df_samples[no_mixture_id][df_samples["y_class"] == y_value]
        )
        raw_indices = np.array(range(len(patients_w_value)))
        raw_indices_shuffled = shuffle(raw_indices, random_state=seed)
        splitted_raw_indices = np.array_split(raw_indices_shuffled, nfolds)
        for f in range(nfolds):
            fold = [patients_w_value[i] for i in splitted_raw_indices[f]]
            db_samples["data_fold" + str(f)] = db_samples["data_fold" + str(f)] + fold

    # creates a dictionary of dataframes, each dataframes holds all records for the fold
    # each ID appears only in one fold
    db = {}
    for f in range(nfolds):
        fold_df = df[df[no_mixture_id].isin(db_samples["data_fold" + str(f)])].copy()
        fold_df["fold"] = f
        db["data_fold" + str(f)] = fold_df
    folds = pd.concat(db, ignore_index=True)
    if print_flag is True:
        print_folds_stat(folds, nfolds, keys_to_balance)
    # remove labels used for creating the partition to folds
    if not debug_mode:
        folds = folds.drop(id_level_labels + record_labels, axis=1)
    return folds


def dataset_balanced_division_to_folds(
    dataset: DatasetBase,
    output_split_filename: str,
    keys_to_balance: Sequence[str],
    nfolds: int,
    id: str = get_sample_id_key(),
    reset_split: bool = False,
    workers: int = 10,
    mp_context: str = None,
    verify_folds_total_size: bool = True,
    **kwargs: dict,
) -> dict:
    """
    Split dataset to folds.
    Support balancing, exclusion and radom seed (with a small improvement could support no mixture criterion).
    :param dataset: FuseMedML style dataset implementation.
    :param output_split_filename: filename to save/read the split from. If the file exist and reset_split=False - this function will return the split stored in reset_split.
    :param keys_to_balance: balancing any possible combination of values. For example for ["data.gender", "data.cancer"], the algorithm will balance each one of the following groups between the folds.
                            (gender=male, cancer=True), (gender=male, cancer=False), (gender=female, cancer=True), (gender=female, cancer=False)

    :param  nfolds : number of folds
    :param  id  : id to balance the split by ( not allowed 2 in same fold)
    :param reset_split: delete output_split_filename and recompute the split
    :param workers : numbers of workers for multiprocessing (export dataset into dataframe)
    :param mp_context : multiprocessing context: "fork", "spawn", etc.
    :param verify_folds_total_size: (bool), when true - if a split file was found, will verify the the number of samples in the folds sum up to the amount of samples in the given dataset
    :param kwargs: more arguments controlling the split. See function balanced_division() for details
    """
    if (
        output_split_filename is not None
        and os.path.exists(output_split_filename)
        and not reset_split
    ):
        folds = load_pickle(output_split_filename)

        # Check that the number of samples in the folds sum up to the dataset size
        dataset_size = len(dataset)
        num_samples_in_folds = sum([len(folds[fold]) for fold in folds])
        if verify_folds_total_size and dataset_size != num_samples_in_folds:
            raise Exception(
                f"Total number of samples founds in split file ({num_samples_in_folds}) is different from the total number of samples in the given dataset ({dataset_size})."
            )
        return folds
    else:
        if id == get_sample_id_key():
            keys = [get_sample_id_key()]
        else:
            keys = [get_sample_id_key(), id]
        if keys_to_balance is not None:
            keys += keys_to_balance
        df = ExportDataset.export_to_dataframe(
            dataset, keys, workers=workers, mp_context=mp_context
        )
        df_folds = balanced_division(df, id, keys_to_balance, nfolds, **kwargs)

        folds = {}
        for fold in range(nfolds):
            folds[fold] = list(df_folds[df_folds["fold"] == fold][get_sample_id_key()])
        if output_split_filename is not None:
            save_pickle(folds, output_split_filename)
        return folds
