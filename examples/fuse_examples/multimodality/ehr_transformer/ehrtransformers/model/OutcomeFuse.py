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

import sys

sys.path.insert(0, "../")

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from datetime import datetime
import ehrtransformers.configs.naming as naming

from ehrtransformers.data_access.utils import (
    detector_idx,
    position_idx,
    seq_pad,
    seq_translate,
)
from ehrtransformers.utils.common import load_pkl
from ehrtransformers.utils.stats import StatsAnalyzer

from fuse.data.datasets.dataset_wrap_seq_to_dict import DatasetWrapSeqToDict
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault


# Set logs/outputs dirs, and save source code relevant to this run


def load_outcome_data(
    file_name,
    dset_name,
    naming_conventions,
    head_config,
    model_config,
    age_month_resolution,
    ageVocab,
    token2idx,
    max_sequence_len,
    batch_size,
    shuffle,
    reverse_inputs=False,
    num_workers=3,
    do_contrastive_sampling=False,
):
    """

    :param file_name: path of the input dataset
    :param dset_name: name of the dataset
    :param naming_conventions: naming conventions dict
    :param head_config: head config class as defined in ehrtransformers/configs/head_config.py
    :param age_month_resolution: 1: age in months, 12: age in years
    :param ageVocab: age vocabulary
    :param token2idx: input (diagnoses, procedures) vocabulary
    :param max_sequence_len: max number of tokens in an input sequence
    :param batch_size: batch size
    :param shuffle: boolean, True - shuffles the data, False - preserves data order
    :param reverse_inputs: boolean, if True the input sequence is reversed, i.e. the last visit is the first input.
        Since the first input embedding is used as the entire patient representation, it makes more sense to make it
        the last visit rather than the first one.
    :return:
    """

    data_dict = load_pkl(file_name)
    data_df = rename_columns_for_behrt(
        data_dict["data_df"],
        naming_conf=naming_conventions,
        age_month_resolution=age_month_resolution,
    )
    # add relevant external GT:
    data_df = head_config.add_external_gt_to_df(data_df, [dset_name])

    stats_inst_tst = StatsAnalyzer(
        data_dict["data_df"],
        naming_conventions,
        title=dset_name,
    )
    stats_inst_tst.print_stats()
    Dset = Outcome(
        head_config=head_config,
        token2idx=token2idx,
        age2idx=ageVocab,
        dataframe=data_df,
        max_len=max_sequence_len,
        reverse_inputs=reverse_inputs,
    )

    out_mapping = [
        "data.age",
        "data.code",
        "data.position",
        "data.segment",
        "data.mask",
        "data.patid",
        "data.vis_date",
    ]  # base output mapping (without heads)
    out_mapping += ["data." + key for key in head_config.get_head_names(only_with_outputs=True)]

    fuse_wrapped_dset = DatasetWrapSeqToDict(name=dset_name, dataset=Dset, sample_keys=out_mapping)
    fuse_wrapped_dset.create()

    if dset_name == "train":
        sampler = BatchSamplerDefault(
            dataset=fuse_wrapped_dset,
            balanced_class_name="data.disease_prediction",
            num_balanced_classes=2,
            batch_size=batch_size,
            mode="approx",
            balanced_class_weights=model_config["sampler_weights"],
            workers=0,
        )
        fuse_dataloader = DataLoader(
            dataset=fuse_wrapped_dset, batch_sampler=sampler, num_workers=num_workers, collate_fn=CollateDefault()
        )
    else:
        fuse_dataloader = DataLoader(
            dataset=fuse_wrapped_dset, batch_size=batch_size, num_workers=num_workers, collate_fn=CollateDefault()
        )
    return data_df, fuse_dataloader  # data_df, dataloader


def cleanup_vocab(token2idx):
    """

    :param token2idx: data token to index dictionary
    :return: a token2idx copy with only data keys (no special keys like "SEP", "CLS", "PAD", "MASK")
    """
    token2idx = token2idx.copy()
    # keys_to_delete = ["PAD", "SEP", "CLS", "MASK"]
    keys_to_delete = [naming.padding_token, naming.separator_token, naming.cls_token, naming.mask_token]
    for k in keys_to_delete:
        del token2idx[k]
    return {x: i for i, x in enumerate(list(token2idx.keys()))}


def date2int(date_YMD):
    return int(date_YMD.strftime("%Y%m%d"))


def int2date(date_YMD_int):
    return datetime.strptime(str(date_YMD_int), "%Y%m%d")


class Outcome(Dataset):
    def __init__(
        self,
        head_config,
        token2idx,
        age2idx,
        dataframe,
        max_len,
        reverse_inputs=False,
    ):
        """
        :param head_config: Head configuration class, as defined in ehrtransformers/configs/head_config.py
        :param token2idx: input dictionary
        :param age2idx: age dictionary
        :param dataframe: input dataframe
        :param max_len: maximal input length (number of tokens)
        :param reverse_inputs:  boolean, if True the input sequence is reversed, i.e. the last visit is the first input.
        Since the first input embedding is used as the entire patient representation, it makes more sense to make it
        the last visit rather than the first one.
        """
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe.code
        self.age = dataframe.age
        self.patid = dataframe.patid
        self.date = dataframe.vis_date
        self.reverse_inputs = reverse_inputs
        self.head_config = head_config
        self.head_inputs = head_config.get_head_inputs(dataframe)
        self.age2idx = age2idx

    def get_inputs_and_aux_encodings(self, index):
        age = self.age[index]
        code = self.code[index]

        # Network inputs:
        if self.reverse_inputs:
            age.reverse()
            code.reverse()
            sep_code = code[0]
            code = code[1:] + [sep_code]

        # get the first self.max_len inputs (code+age), set the first input to be 'CLS'
        age = age[(-self.max_len + 1) :]
        code = code[(-self.max_len + 1) :]
        if (code[0] != naming.separator_token) and (len(code) < self.max_len):
            code = np.append(np.array([naming.cls_token]), code)
            age = np.append(np.array(age[0]), age)
        else:
            code[0] = naming.cls_token

        # mask
        mask = np.ones(self.max_len)
        mask[len(code) :] = 0

        tokens, code = seq_translate(code, self.vocab)

        # position and segment encodings
        tokens = seq_pad(tokens, self.max_len)
        position = position_idx(tokens)
        segment = detector_idx(tokens)

        # pad ages and codes to length self.max_len
        code = seq_pad(code, self.max_len, symbol=self.vocab[naming.padding_token])
        _, age = seq_translate(age, translate_dict=self.age2idx)
        age = seq_pad(age, self.max_len, symbol=age[-1])
        return code, age, position, segment, mask

    def __getitem__(self, index):
        """

        :param index:
        :return: code, age, position (preparation for BEHRT, unused in Bert), segment (preparation for BEHRT, unused in Bert), mask
        """

        patid = self.patid[index]
        vis_date = self.date[index]

        code, age, position, segment, mask = self.get_inputs_and_aux_encodings(index)
        # Ground Truth:
        output_values = {head_name: self.head_inputs[head_name][index] for head_name in self.head_inputs}
        try:
            translated_output_values = tuple(
                self.head_config.translate_input_value(head_name, output_values[head_name])
                for head_name in output_values
            )
        except Exception as e:
            raise e

        return_tuple = (
            torch.LongTensor(age),  # len max_len_seq
            torch.LongTensor(code),  # len max_len_seq
            torch.LongTensor(position),  # len max_len_seq
            torch.LongTensor(segment),  # len max_len_seq
            torch.LongTensor(mask),  # len max_len_seq
            torch.LongTensor([int(patid)]),  # len 1
            torch.LongTensor([date2int(vis_date)]),  # len 1
        )

        return return_tuple + translated_output_values

    def __len__(self):
        return len(self.code)


def rename_columns_for_behrt(df, naming_conf, age_month_resolution):
    """
    Replaces column names of input df (as defined in naming_config) with names expected by BEHRT (TODO: make this configurable too)
    :param df: input patient EHR dataframe
    :param naming_conf: column names of input df
    :param global_params: main parameter of interest is global_params['month'], which means age month resolution. when 1 means the age will be in months, and when 12 - in years
    :return: df with relevant columns renamed
    TODO: move this function to a separate import, since it may be used by different modules

    naming_config (for reference,
    defined in ehrtransformers.configs.config) =
    'diagnosis_vec_key': 'DX',
    'age_key': 'AGE',
    'age_month_key': 'AGE_MON',
    'label_key': 'label',
    'date_key': "ADMDATE",
    'patient_id_key': "ENROLID",
    'outcome_key': "GT",
    'event_key': "EVENTS"
    'gender_key': "SEX",
    'separator_str' : 'SEP',
    'sick_val' : '1',
    'healthy_val' : '0'

    """
    # Vadim start: column names in the DB to replace
    diagnosis_vec_key_in = naming_conf["diagnosis_vec_key"]  #'DX'
    diagnosis_vec_key_out = "code"
    label_key_in = naming_conf["event_key"]
    label_key_out = "label"
    treatment_key_in = naming_conf["treatment_event_key"]
    treatment_key_out = "treatment_event"
    patient_id_key_in = naming_conf["patient_id_key"]  # "ENROLID"
    patient_id_key_out = "patid"

    gender_key_in = naming_conf["gender_key"]
    gender_key_out = "gender"

    next_visit_key_in = naming_conf["next_visit_key"]
    next_visit_key_out = "next_visit"

    date_key_in = naming_conf["date_key"]
    date_key_out = "vis_date"
    if age_month_resolution == 1:
        age_key_in = "AGE_MON"
    elif age_month_resolution == 12:
        age_key_in = "AGE"
    else:
        raise Exception(
            "Unexpected age month resolution (should be 1/12 for months/years) {}".format(age_month_resolution)
        )
    age_key_out = "age"
    data_column_renames = {
        diagnosis_vec_key_in: diagnosis_vec_key_out,
        age_key_in: age_key_out,
        label_key_in: label_key_out,
        patient_id_key_in: patient_id_key_out,
        date_key_in: date_key_out,
        gender_key_in: gender_key_out,
        next_visit_key_in: next_visit_key_out,
        treatment_key_in: treatment_key_out,
    }

    out_df = df.rename(columns=data_column_renames, inplace=False)

    return out_df
