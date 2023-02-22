import pickle
from collections import OrderedDict
from random import randrange
from typing import Sequence, Any, Tuple, List

import glob
import os
import numpy as np
import pandas as pd

from fuse.data import DatasetDefault, PipelineDefault, OpBase
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_common import OpLookup, OpSetIfNotExist
from fuse.utils.ndict import NDict

from fuse.data.utils.split import dataset_balanced_division_to_folds
from fuse.data.utils.export import ExportDataset

from fuse_examples.multimodality.ehr_transformer.utils import (
    seq_translate,
    position_idx,
    special_tokens,
    seq_pad,
    WordVocab,
)


VALID_TESTS_ABOVE_ZERO = [
    "pH",
    "Weight",
    "Height",
    "DiasABP",
    "HR",
    "NIMAP",
    "MAP",
    "NIDiasABP",
    "NISysABP",
    "PaCO2",
    "PaO2",
    "Temp",
    "SaO2",
    "RespRate",
    "SysABP",
]

STATIC_FIELDS = ["Age", "Gender", "Height", "ICUType", "Weight"]


class OpAddBMI(OpBase):

    """
    Apply BMI calculation using static patient's fields. Names of the fields
    are provided as parameters as well as output field name.
    Height expected to be in centimeters
    Weight expected to be in kilograms

    Example:
    OpAddBMI(key_in_height="Height", key_in_weight="Weight", key_out_bmi="BMI")
    """

    def __init__(self, key_in_height: str, key_in_weight: str, key_out_bmi: str):

        super().__init__()
        self._key_height = key_in_height
        self._key_weight = key_in_weight
        self._key_bmi = key_out_bmi

    def __call__(self, sample_dict: NDict) -> Any:

        if (self._key_height in sample_dict["StaticDetails"]) & (self._key_weight in sample_dict["StaticDetails"]):
            height = sample_dict["StaticDetails." + self._key_height]
            weight = sample_dict["StaticDetails." + self._key_weight]
            if ~np.isnan(height) & ~np.isnan(weight):
                sample_dict["StaticDetails." + self._key_bmi] = 10000 * weight / (height * height)

        return sample_dict


class OpConvertVisitToSentence(OpBase):
    """
    Converts the observations presented in data frame
     to sentence by visit time. Sentence will be presented as
     list of values per visit (converted to categorical strings)
     and separator at the end of the sentence
    """

    def __call__(self, sample_dict: NDict) -> Any:

        df_visits = sample_dict["Visits"]

        d_visit_sentences = OrderedDict()

        for visit_time, df_visit in df_visits.groupby("Time", sort=True):

            d_visit_sentences[visit_time] = []

            d_visit_sentences[visit_time].extend(df_visit["Value"].to_list())
            d_visit_sentences[visit_time].extend(["SEP"])

        sample_dict["VisitSentences"] = d_visit_sentences

        return sample_dict


class OpGenerateFinalTrajectoryOfVisits(OpBase):
    """
    Generates final trajectory of visits before training including:
        - random extraction of sub-trajectory (as augmentation procedure)
        - limitation its size by configured max_len of a sequence
        - embedding static features, according to defined configuration
        - adding special tokens of classifier and padding to the final sequence
        - translating tokens to indexes
        - computation of positional indexes of sentences for further embedding
    """

    def __init__(self, max_len: int, vocab: dict, embed_static_in_all_visits: int, static_variables_to_embed: list):

        super().__init__()
        self._max_len = max_len
        self._vocab = vocab
        self._static_variables_to_embed = static_variables_to_embed
        self._embed_static_in_all_visits = embed_static_in_all_visits

    def __call__(self, sample_dict: NDict) -> Any:

        d_visits_sentences = sample_dict["VisitSentences"]
        n_visits = len(d_visits_sentences)

        # extract sub-set of trajectory by random selection of
        # start and stop keys we use n_visits- 1 to keep the last
        # visit for outcome in classifier
        start_visit = randrange(int((n_visits - 1) / 2))
        stop_visit = n_visits - randrange(int((n_visits - 1) / 2)) - 1
        keys = list(d_visits_sentences.keys())
        trajectory_keys = keys[start_visit:stop_visit]
        next_visit = keys[stop_visit]

        # add static variables as a first special visit
        df_static = sample_dict["StaticDetails"]
        static_embeddings = []
        if len(self._static_variables_to_embed) > 0:
            for k in df_static.keys():
                if k in self._static_variables_to_embed:
                    static_embeddings += [df_static[k]]

        # Build trajectory of visits
        # appends sentences in the reverse order from
        # stop to start in order to get X last sentences that
        # less than max_len
        trajectory_tokens = []
        for k in reversed(trajectory_keys):
            visit_sentence = d_visits_sentences[k]

            # add static details to every visit
            if self._embed_static_in_all_visits:
                visit_sentence = static_embeddings + visit_sentence

            if (len(trajectory_tokens) + len(visit_sentence)) < self._max_len:
                trajectory_tokens = visit_sentence + trajectory_tokens
            else:
                break

        # create final sequence of tokens
        tokens = [special_tokens["cls"]]
        if not self._embed_static_in_all_visits:
            static_embeddings += [special_tokens["separator_static"]]
            tokens += static_embeddings
        tokens += trajectory_tokens

        tokens = seq_pad(tokens, self._max_len)
        positions = position_idx(tokens)
        token_ids = seq_translate(tokens, self._vocab)

        sample_dict["Tokens"] = tokens
        sample_dict["Positions"] = positions
        sample_dict["Indexes"] = np.array(token_ids).squeeze(0)

        # outcomes of next visit prediction
        sample_dict["NextVisitTokens"] = d_visits_sentences[next_visit]
        sample_dict["NextVisitTokenIds"] = seq_translate(sample_dict["NextVisitTokens"], self._vocab)
        sample_dict["NextVisitLabels"] = np.zeros(len(self._vocab))
        sample_dict["NextVisitLabels"][sample_dict["NextVisitTokenIds"]] = 1.0

        return sample_dict


class OpMapToCategorical(OpBase):
    """
    Converts continuous values of observations
    to the category based on calculated "bins"
    each pair of name + value will be converted
    to the word "Name_binN"

     Example: Age = 80 will be converted to "Age_3"
      (third bin in calculated distribution)
    """

    def __call__(self, sample_dict: NDict, bins: dict) -> Any:

        # convert continuous measurements to categorical ones based
        # on defined bins mapping static clinical characteristics
        # (Age, Gender, ICU type, Height, etc)
        for k in sample_dict["StaticDetails"]:
            sample_dict[f"StaticDetails.{k}"] = k + "_" + str(np.digitize(sample_dict["StaticDetails"][k], bins[k]))

        # mapping labs exams and clinical characteristics captured
        # during patients' stay in ICU
        if not sample_dict["Visits"].empty:
            sample_dict["Visits"]["Value"] = sample_dict["Visits"].apply(
                lambda row: row["Parameter"] + "_" + str(np.digitize(row["Value"], bins[row["Parameter"]]))
                if row["Parameter"] in bins
                else special_tokens["unknown"],
                axis=1,
            )

        return sample_dict


class PhysioNetCinC:
    @staticmethod
    def _read_raw_data(raw_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads raw data from the provided directory path.
         It includes clinical patients data and outcomes
        :param raw_data_path: path to folder with raw data
        :return: data frame of patients observation records
        (record per observation) and data frame of outcomes
         (record per patient)
        """
        # read patients info and lab exams
        df = pd.DataFrame(columns=["PatientId", "Time", "Parameter", "Value"])
        data_sub_sets = ["set-a", "set-b"]
        for s in data_sub_sets:
            csv_files = glob.glob(os.path.join(raw_data_path + "/" + s, "*.txt"))
            for f in csv_files:
                patient_id = os.path.splitext(os.path.basename(f))[0]
                df_file = pd.read_csv(f)
                df_file = df_file.drop(df_file[(df_file["Parameter"] == "RecordID")].index).reset_index(drop=True)
                df_file["PatientId"] = patient_id
                df = pd.concat([df, df_file], axis=0)
        df.reset_index(inplace=True, drop=True)
        patient_ids = np.unique(df["PatientId"].values)

        # read outcomes
        df_outcomes = pd.DataFrame(columns=["RecordID", "In-hospital_death"])
        outcomes = ["Outcomes-a.txt", "Outcomes-b.txt"]
        for o in outcomes:
            o_file = os.path.join(raw_data_path + "/" + o)
            df_outcomes = df_outcomes.append(pd.read_csv(o_file)[["RecordID", "In-hospital_death"]]).reset_index(
                drop=True
            )
        df_outcomes["RecordID"] = df_outcomes["RecordID"].astype(str)
        df_outcomes.rename(columns={"RecordID": "PatientId"}, inplace=True)

        # synchronize with patients data
        df_outcomes = df_outcomes[df_outcomes["PatientId"].isin(patient_ids)]

        return df, df_outcomes

    @staticmethod
    def _drop_records_with_errors(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop records with illegal values of observations
        :param df: data frame with observational records
        :return: updated data frame of patients' observations
        """
        # drop records with measurements below or equal
        # to zero for tests with only positive values
        for v in VALID_TESTS_ABOVE_ZERO:
            df = df.drop(df[(df["Parameter"] == v) & (df["Value"] <= 0)].index).reset_index(drop=True)

        # drop gender values below zero
        df = df.drop(df[(df["Parameter"] == "Gender") & (df["Value"] < 0)].index).reset_index(drop=True)

        # drop records with invalid Height and Weight values
        df = df.drop(df[(df["Parameter"] == "Weight") & (df["Value"] < 20)].index).reset_index(drop=True)
        df = df.drop(df[(df["Parameter"] == "Height") & (df["Value"] < 100)].index).reset_index(drop=True)
        return df

    @staticmethod
    def _drop_bad_patients(df: pd.DataFrame, min_hours: int, min_num_of_visits: int) -> pd.DataFrame:
        """
        Drop records of patients that have the less than
        min_hours reported in ICU and  patient that have less
         than minimum number of visits
        :param df:
        :param min_hours: minimum number of hours of taken observations
        :param min_num_of_visits: minimum number of
        visits (observations taking events)
        :return: updated data frame of patients' observation records
        """

        df_fixed = df.copy()
        count_dropped_short_time = 0
        count_dropped_min_visits = 0
        for pat_id, df_pat_records in df.groupby("PatientId"):

            hours = df_pat_records["Time"].str.split(":", 1, True)[0].values
            if max(hours.astype(int)) < min_hours:
                df_fixed.drop(df_pat_records.index, inplace=True)
                count_dropped_short_time += 1
                continue

            num_visits = len(np.unique(df_pat_records["Time"].values))
            if num_visits < min_num_of_visits:
                df_fixed.drop(df_pat_records.index, inplace=True)
                count_dropped_min_visits += 1

        return df_fixed

    @staticmethod
    def _combine_data_by_patients(df_records: pd.DataFrame, df_outcomes: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all available information : static details,
        visits and target outcome in one data frame of patients
        :param df_records: data frame of observation records
        :param df_outcomes: date frame of patients' outcomes
        :return: combined data frame of patients
        """

        static_fields = ["Age", "Height", "Weight", "Gender", "ICUType"]

        patient_ids = []
        static_details = []
        visits = []
        target = []
        for pat_id, df_pat_records in df_records.groupby("PatientId"):
            patient_ids += [pat_id]
            # extract static patients details
            static_info_ind = (df_pat_records["Time"] == "00:00") & (df_pat_records["Parameter"].isin(static_fields))
            df_pat_static = df_pat_records[static_info_ind]
            static_details += [dict(zip(df_pat_static.Parameter, df_pat_static.Value))]

            # extract visits (time series events)
            df_pat_dynamic_exams = df_pat_records[~static_info_ind]
            visits += [df_pat_dynamic_exams.copy()]

            label = df_outcomes[df_outcomes["PatientId"] == pat_id]["In-hospital_death"].values[0]
            target += [label]

        list_of_tuples = list(zip(patient_ids, static_details, visits, target))
        df_patients = pd.DataFrame(list_of_tuples, columns=["PatientId", "StaticDetails", "Visits", "Target"])

        return df_patients

    @staticmethod
    def _generate_percentiles(
        dataset: DatasetDefault, num_percentiles: int, categorical_max_num_of_values: int
    ) -> dict:
        """

        :param dataset: data frame of observational records
        :param num_percentiles: number of bins to digitize the
         continuous observations
        :param categorical_max_num_of_values: which observations can be
        counted as categorical based on number
        of unique values.
        :return: dictionary of calculated percentiles (bins)
        """

        df = ExportDataset.export_to_dataframe(dataset, keys=["StaticDetails", "Visits"], workers=1)

        # Extracting static and dynamic parts of the dataset
        df_static = pd.DataFrame(df["StaticDetails"].to_list())
        df_visits = pd.concat(df["Visits"].values)

        # generation dictionaries of values for static
        # and dynamic variables of dataset patients
        d_static = df_static.to_dict("list")
        d_visits = dict.fromkeys(np.unique(df_visits[["Parameter"]]))
        for k in d_visits.keys():
            d_visits[k] = df_visits[df_visits["Parameter"] == k]["Value"]

        d_all_values = d_visits
        d_all_values.update(d_static)

        # calculate percentiles for categorical parameters
        # (Gender, etc) update percentiles according to categories
        d_percentile = dict()
        percentiles = range(0, 100 + int(100 / num_percentiles), int(100 / num_percentiles))
        for k in d_all_values.keys():
            values = np.array(d_all_values[k])
            values = values[~np.isnan(values)]
            # check number of unique values
            unique_values = set(values)
            if len(unique_values) < categorical_max_num_of_values:
                # categorical value
                unique_values = sorted(unique_values)
                # incrementing in 1 is needed for getting bin number
                # corresponding to the variable value
                # e.g. gender values will remain the same
                unique_values = [x + 1 for x in unique_values]
                d_percentile[k] = sorted(unique_values)
            else:
                d_percentile[k] = np.percentile(values, percentiles)

        return d_percentile

    @staticmethod
    def _build_corpus_of_words(dict_percentiles: dict) -> list:
        """
        Generates a list of possible words that will be generated
         by mapping to bins. This list will define a corpus to the
          final vocabulary
        :param dict_percentiles: dictionary with calculated
        percentiles (bins) for each observation type
        :return: corpus of words
        """

        corpus = list(special_tokens.values())

        for k in dict_percentiles.keys():
            num_bins = len(dict_percentiles[k])
            # Create words for all possible bins in percentile
            for b in range(0, num_bins + 1):
                word = k + "_" + str(b)
                corpus += [word]

        return corpus

    @staticmethod
    def _load_and_process_df(
        raw_data_path: str, raw_data_pkl: str, min_hours_in_hospital: int, min_number_of_visits: int
    ) -> Tuple[pd.DataFrame, list]:
        """
        Applies all stages for generation patients' data frame.
        Manages generated data frame in pickle file.

        :param raw_data_path: path to the raw data folder
        :param raw_data_pkl: path to the pickle file with generated
        patients data frame
        :param min_hours_in_hospital: used for filtering illegal patients
        :param min_number_of_visits: used for filtering illegal patients
        :return: data frame of patients and list of patient ids
        """
        # if pickle available
        try:
            df_patients, patient_ids = pickle.load(open(raw_data_pkl, "rb"))

        except Exception:
            df_raw_data, df_outcomes = PhysioNetCinC._read_raw_data(raw_data_path)

            # drop records with invalid tests results
            df_raw_data = PhysioNetCinC._drop_records_with_errors(df_raw_data)

            # drop patients with less than minimum hours in hospital and less
            # than minimum number of visits
            df_raw_data = PhysioNetCinC._drop_bad_patients(df_raw_data, min_hours_in_hospital, min_number_of_visits)
            patient_ids = np.unique(df_raw_data["PatientId"].values)

            df_patients = PhysioNetCinC._combine_data_by_patients(df_raw_data, df_outcomes)

            if raw_data_pkl is not None:
                with open(raw_data_pkl, "wb") as f:
                    pickle.dump([df_patients, patient_ids], f)

        return df_patients, patient_ids

    @staticmethod
    def _process_dynamic_pipeline(
        dict_percentiles: dict,
        static_variables_to_embed: Any,
        embed_static_in_all_visits: Any,
        token2idx: Any,
        max_len: int,
    ) -> List[OpBase]:
        """
        Generates dynamic pipeline by stacking a list of Operators.
        :param dict_percentiles: percentiles of observations
        :param static_variables_to_embed: the list of static variables
        to embed in patients trajectory
        :param embed_static_in_all_visits: the way of embedding static
        variables
        :param token2idx: vocabulary of tokens to indexes
        :param max_len: maximal lengths of tokens' sequence
        :return: list of Operators
        """
        return [
            (OpMapToCategorical(), dict(bins=dict_percentiles)),
            (OpConvertVisitToSentence(), dict()),
            (
                OpGenerateFinalTrajectoryOfVisits(
                    max_len, token2idx, embed_static_in_all_visits, static_variables_to_embed
                ),
                dict(),
            ),
            (OpSetIfNotExist(), dict(key="StaticDetails.Gender", value=-1)),
            (
                OpLookup(map={"Gender_0": 0, "Gender_1": 1, -1: -1}),
                dict(key_in="StaticDetails.Gender", key_out="Gender"),
            ),
        ]

    @staticmethod
    def dataset(
        raw_data_path: str,
        raw_data_pkl: str,
        num_folds: int,
        split_filename: str,
        seed: int,
        reset_split: bool,
        train_folds: Sequence[int],
        validation_folds: Sequence[int],
        test_folds: Sequence[int],
        num_percentiles: int,
        categorical_max_num_of_values: int,
        min_hours_in_hospital: int,
        min_number_of_visits: int,
        static_variables_to_embed: Sequence[str],
        embed_static_in_all_visits: int,
        max_len_seq: int,
    ) -> Tuple[Any, DatasetDefault, DatasetDefault, DatasetDefault]:
        """
        This is the main methods for dataset generation. It includes the stages
         - raw data loading and processing for generation data frame with
         patients' records
         - preparation percentiles for patients observations mapping (on train)
         - split to train/validation/test
         - creation of dynamic pipline in two stages:
            first stage (before percentile calculation) will include loading
            data to sample-dicts and calculation additional observations (BMI)
            second stage (after percentiles calculation) will include mapping
            to categorical using percentiles as bins; converting observations
            to sentences and generation random sub-trajectories using this
            sentences, static details will be embedded in trajectories.
         - generation train, validation and test datasets with defined
            dynamic pipline

        below input parameters are defined in config,yaml in dataset group

        :param raw_data_path: path to raw data folder
        :param raw_data_pkl: path to the pickle file with loaded data
        :param num_folds: number of folds
        :param split_filename: file of splits
        :param seed: random seed
        :param reset_split: bool for split reset
        :param train_folds: indexes of train folds
        :param validation_folds: indexes of validation folds
        :param test_folds: indexes of test folds
        :param num_percentiles: number of percentiles for mapping continuous
         observations
        :param categorical_max_num_of_values: definition of categorical
        in observations
        :param min_hours_in_hospital: minimum allowed numbers of hours
        in hospital
        :param min_number_of_visits: minimum allowed numbers of visits
        :param static_variables_to_embed: list of static variables for
        embedding
        :param embed_static_in_all_visits: way of embedding static  details
        :param max_len_seq: maximal length of token sequence
        :return: vocabulary of tokens to indexes, datasets of train,
        validation and test
        """
        assert raw_data_path is not None

        df_patients, patient_ids = PhysioNetCinC._load_and_process_df(
            raw_data_path, raw_data_pkl, min_hours_in_hospital, min_number_of_visits
        )

        # first step of the pipline is to read data and add
        # additional features (e.g. BMI)
        dynamic_pipeline_ops = [
            (OpReadDataframe(df_patients, key_column="PatientId"), {}),
            (OpAddBMI(key_in_height="Height", key_in_weight="Weight", key_out_bmi="BMI"), dict()),
        ]
        dynamic_pipeline = PipelineDefault("cinc_dynamic", dynamic_pipeline_ops)

        dataset_all = DatasetDefault(patient_ids, dynamic_pipeline)
        dataset_all.create()

        folds = dataset_balanced_division_to_folds(
            dataset=dataset_all,
            output_split_filename=split_filename,
            keys_to_balance=["Target"],
            nfolds=num_folds,
            seed=seed,
            reset_split=reset_split,
            workers=1,
        )

        train_sample_ids = []
        for fold in train_folds:
            train_sample_ids += folds[fold]
        dataset_train = DatasetDefault(train_sample_ids, dynamic_pipeline)
        dataset_train.create()

        # calculate statistics of train set only and generate dictionary
        # of percentiles for mapping lab results to categorical for train,
        # validation and test
        dict_percentiles = PhysioNetCinC._generate_percentiles(
            dataset_train, num_percentiles, categorical_max_num_of_values
        )

        # generation corpus and tokenizer
        corpus = PhysioNetCinC._build_corpus_of_words(dict_percentiles)
        token2idx = WordVocab(corpus, max_size=None, min_freq=1).get_stoi()

        # update pipeline with Op using calculated percentiles
        dynamic_pipeline_ops = dynamic_pipeline_ops + [
            *PhysioNetCinC._process_dynamic_pipeline(
                dict_percentiles, static_variables_to_embed, embed_static_in_all_visits, token2idx, max_len_seq
            )
        ]
        dynamic_pipeline = PipelineDefault("cinc_dynamic", dynamic_pipeline_ops)

        # update dynamic pipeline after calculating general statistics
        # and percentiles based on train dataset
        dataset_train._dynamic_pipeline = dynamic_pipeline

        validation_sample_ids = []
        for fold in validation_folds:
            validation_sample_ids += folds[fold]
        dataset_validation = DatasetDefault(validation_sample_ids, dynamic_pipeline)
        dataset_validation.create()

        test_sample_ids = []
        for fold in test_folds:
            test_sample_ids += folds[fold]
        dataset_test = DatasetDefault(test_sample_ids, dynamic_pipeline)
        dataset_test.create()

        return token2idx, dataset_train, dataset_validation, dataset_test


if __name__ == "__main__":
    token2idx, ds_train, ds_valid, ds_test = PhysioNetCinC.dataset(
        os.environ["CINC_DATA_PATH"],
        None,
        5,
        None,
        1234,
        True,
        [0, 1, 2],
        [3],
        [4],
        4,
        5,
        46,
        10,
        ["Age", "Gender", "BMI"],
        0,
        350,
    )
