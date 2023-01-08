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

import random
from pathlib import Path
import json
import pandas as pd
import ehrtransformers.configs.naming as naming


def seq_translate(tokens, translate_dict, mask_token=None):
    """
    returns a list of tokens translated using translate_dict
    :param tokens:
    :param translate_dict:
    :param mask_token: special token that is always translated to 'UNK'
    :return:
    """
    return (
        tokens,
        [
            translate_dict.get(token, translate_dict[naming.unknown_token])
            if token != mask_token
            else translate_dict[naming.unknown_token]
            for token in tokens
        ],
    )


def load_icd_to_ccs() -> dict:
    """Load the icd to ccs mapping supplied by the package.



    Returns:
        dict: a mapping of icd code to ccs code
    """
    fname = Path(__file__).parent / "icd_to_ccs.json"
    with open(fname) as f:
        return json.load(f)


def reduce_vocab(token2idx: dict, token_map: dict) -> dict:
    """Reduce vocabulary by utilizing a new token mapping.

    The tokens in token_map are assumed to map from many to one such that the
    final vocabulary size will be smaller. `token_map` is assumed to contain
    all of the same keys as token2idx and potentially many more, except for the
    special tokens: {"PAD", "UNK", "SEP", "CLS", "MASK"}. The special tokens
    will have the first indices in the new reduced vocab as well. Note that the
    new vocab will likely NOT by 1:1 and as such idx2token does not exist.


    Args:
        token2idx (dict): token2idx mapping (eg, from icd code to integer)
        token_map (dict): token mapping to another representation, for example
            from icd code to ccs code. The values are NOT assumed to be integer
            or unique.

    Returns:
        dict: a new token2idx dict with the same set of keys as the original
            token2idx but with indices based on the mapping `token_map`. If a token
            does not have a token_map value, it's mapped to 'UNK' mapping.
    """
    specials = [naming.padding_token, naming.separator_token, naming.unknown_token, naming.cls_token, naming.mask_token]
    # remap_used = {k: token_map.get[k] if k in token_map else k for k in token2idx} #This way, if we have unknown tokens that are not specials, they are left as is, and cause an error
    remap_used = {k: token_map.get(k, naming.unknown_token) if k not in specials else k for k in token2idx}

    uniques = set(remap_used.values())
    ordered_uniques = sorted(uniques & specials) + sorted(uniques - specials)
    value_to_idx = {k: idx for idx, k in enumerate(ordered_uniques)}
    reduced_vocab = {k: value_to_idx[v] for k, v in remap_used.items()}
    return reduced_vocab


def detector_idx(tokens, symbol=naming.separator_token):
    """
    Given a sequence of codes divided into groups (visits) by symbol ('SEP') tokens, returns a sequence of the same
    size of binary visit indicators, so that codes from a single visit get the same indicator, and adjacent visits -
    different indicators.
    :param tokens:
    :param symbol:
    :return:
    """
    flag = 0
    detector = []

    for token in tokens:
        detector.append(flag)
        if token == symbol:
            flag = 1 - flag
    return detector


def position_idx(tokens, symbol=naming.separator_token):
    """
    Given a sequence of codes divided into groups (visits) by symbol ('SEP') tokens, returns a sequence of the same
    size of visit indices.
    :param tokens:
    :param symbol:
    :return:
    """
    group_inds = []
    flag = 0
    for token in tokens:
        group_inds.append(flag)
        if token == symbol:
            flag += 1
    return group_inds


def seq_pad(tokens, max_len, symbol=naming.padding_token):
    """
    Returns a list of tokens padded by symbol to length max_len.
    :param tokens:
    :param max_len:
    :param token2idx:
    :param symbol:
    :return:
    """
    token_len = len(tokens)
    if token_len < max_len:
        return list(tokens) + [symbol] * (max_len - token_len)

    else:
        return tokens[:max_len]


class DataAdder:
    source_keys = None
    target_key = None
    mult_comb_mode = None
    source_on_keys = None
    target_on_keys = None

    def __init__(
        self,
        source_data_dict,
        source_keys,
        target_key,
        source_on_keys,
        target_on_keys,
        mult_source_comb_mode=None,
        val_fillna=None,
    ):
        """

        :param source_data_dict: dictionary of set:path:
            {
            'train': '/path_to_train_GT/' or None
            'val': '/path_to_val_GT/' or None
            'test': '/path_to_test_GT/' or None
            }
            the path points to a file that contains the GT data (table or df). If None - the path is ignored
        :param source_keys: columns from which to extract the data that will be added to the target df
        :param target_key: name of the column in target df to which data will be added from source df
        :param source_on_keys: on which source df keys to merge the data
        :param target_on_keys: on which target keys to merge the data
        :param mult_source_comb_mode:
        """
        if isinstance(source_keys, list):
            self.source_keys = source_keys
        elif isinstance(source_keys, str):
            if source_keys == "all":
                self.source_keys = "all"
            else:
                self.source_keys = [source_keys]
        else:
            raise Exception("unexpected source_keys")
        self.target_key = target_key
        self.mult_comb_mode = mult_source_comb_mode
        self.source_on_keys = source_on_keys
        self.target_on_keys = target_on_keys
        self.source_data_dict = source_data_dict
        self.val_fillna = val_fillna

    def get_num_classes(self):
        if isinstance(self.source_keys, list):
            return len(self.source_keys)
        elif self.source_keys == "all":
            all_keys = set([])
            set_paths = []
            for set_name in self.source_data_dict:
                set_name = set_name.lower()
                if "val" in set_name:
                    if self.source_data_dict["val"] is not None:
                        set_paths.append(self.source_data_dict["val"])
                elif "tr" in set_name:
                    if self.source_data_dict["train"] is not None:
                        set_paths.append(self.source_data_dict["train"])
                elif "st" in set_name:
                    if self.source_data_dict["test"] is not None:
                        set_paths.append(self.source_data_dict["test"])

            set_paths = list(set(set_paths))  # remove duplicate paths

            # read source dataframes:
            source_df_list = []
            for set_fname in set_paths:
                if isinstance(set_fname, str):
                    if set_fname.endswith(".csv"):
                        temp_df = pd.read_csv(set_fname)
                    else:
                        raise Exception("not implemented for non-csv tables")
                else:
                    raise Exception("not implemented for non-pathnames")
                all_keys = all_keys.union(set(temp_df.columns))
            for k in self.source_on_keys:
                if k in all_keys:
                    all_keys.remove(k)
            return len(all_keys)

    def combine(self, main_df, set_names):
        """

        :param main_df: df to which we want to add the GT column
        :param set_names: a list containing any of 'train', 'val', or 'test'. name of the set the df describes. If None - we add merged train/val/test external datasets
        :return:
        """
        # read pathnames:
        if isinstance(set_names, str):
            set_names = [set_names]
        if set_names is None:
            set_names = ["train", "val", "test"]
        set_paths = []
        all_keys = set([])
        for i, set_name in enumerate(set_names):
            set_name = set_name.lower()
            if "val" in set_name:
                if self.source_data_dict["val"] is not None:
                    set_paths.append(self.source_data_dict["val"])
            elif "tr" in set_name:
                if self.source_data_dict["train"] is not None:
                    set_paths.append(self.source_data_dict["train"])
            elif "st" in set_name:
                if self.source_data_dict["test"] is not None:
                    set_paths.append(self.source_data_dict["test"])

        set_paths = list(set(set_paths))  # remove duplicate paths

        # read source dataframes:
        source_df_list = []
        for set_fname in set_paths:
            if isinstance(set_fname, str):
                if set_fname.endswith(".csv"):
                    temp_df = pd.read_csv(set_fname)
                else:
                    raise Exception("not implemented for non-csv tables")
            else:
                raise Exception("not implemented for non-pathnames")
            source_df_list.append(temp_df)
            all_keys = all_keys.union(set(temp_df.columns))

        if len(source_df_list) == 0:
            return None

        comb_df_source = pd.concat(source_df_list, axis=0).drop_duplicates().reset_index(drop=True)
        # combine source columns into a single column:
        if isinstance(self.source_keys, str):
            if self.source_keys.lower() == "all":
                for k in self.source_on_keys:
                    if k in all_keys:
                        all_keys.remove(k)
                self.source_keys = list(all_keys)
            else:
                raise Exception("Unexpected source keys")

        if len(self.source_keys) > 1:
            # If there are multiple keys - the new target_key column will contain a list of their values
            comb_df_source[self.target_key] = comb_df_source[self.source_keys].values.tolist()
        else:
            comb_df_source[self.target_key] = comb_df_source[self.source_keys[0]]

        # leave only source_on_keys and target_key:
        comb_df_source = comb_df_source.loc[
            :, comb_df_source.columns.intersection(self.source_on_keys + [self.target_key])
        ]

        # merge source and target df's:
        out_df = main_df.merge(
            right=comb_df_source,
            how="left",
            left_on=self.target_on_keys,
            right_on=self.source_on_keys,  #'inner',
        )
        if self.val_fillna != None:
            if len(self.source_keys) > 1:
                out_df[self.target_key] = out_df[self.target_key].apply(
                    lambda x: x if isinstance(x, list) else [self.val_fillna] * len(self.source_keys)
                )
            else:
                out_df = out_df.fillna({self.target_key: self.val_fillna})
        return out_df
