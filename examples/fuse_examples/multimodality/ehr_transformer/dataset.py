import pickle
from collections import OrderedDict
from random import randrange
from typing import Sequence, Any
import re
import glob
import os

import numpy as np
import pandas as pd
from typing import Tuple

from fuse.data import DatasetDefault, PipelineDefault, OpBase
from fuse.data.ops.ops_read import OpReadDataframe
# from fuse.data.ops.ops_common import OpCond, OpSet
from fuse.data.utils.split import dataset_balanced_division_to_folds
from ops_read_cinc import OpReadDataframeCinC
from fuse.data.utils.export import ExportDataset

SOURCE = r'C:/D_Drive/Projects/EHR_Transformer/PhysioNet/predicting-mortality-of-icu-patients-the-physionetcomputing-in-cardiology-challenge-2012-1.0.0/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0'

VALID_TESTS_ABOVE_ZERO = ['pH', 'Weight', 'Height', 'DiasABP', 'HR', 'NIMAP', 'MAP', 'NIDiasABP', 'NISysABP', 'PaCO2',
                          'PaO2', 'Temp', 'SaO2', 'RespRate', 'SysABP']

STATIC_FIELDS = ['Age', 'Gender', 'Height', 'ICUType', 'Weight']


class OpAddBMI(OpBase):
    def __call__(self, sample_dict) -> Any:

        d_static = sample_dict['StaticDetails']

        if ("Height" in d_static.keys()) & ("Weight" in d_static.keys()):
            height = d_static["Height"]
            weight = d_static["Weight"]
            if ~np.isnan(height) & ~np.isnan(weight):
                d_static["BMI"] = 10000 * weight / (height * height)

        sample_dict['StaticDetails'] = d_static
        return sample_dict


class OpConvertVisitToSentense(OpBase):
    def __call__(self, sample_dict, visit_to_embed_static_variables, static_variables_to_embed) -> Any:

        df_static = sample_dict['StaticDetails']
        df_visits = sample_dict['Visits']
        # convert statis details for embedding to list
        static_embeddings = []
        if visit_to_embed_static_variables is not None:
            for k in df_static.keys():
                if k in static_variables_to_embed:
                    static_embeddings += [df_static[k]]

        d_visit_sentences = OrderedDict()
        first_visit_embedded = False

        for visit_time, df_visit in df_visits.groupby('DateTime', sort=True):

            d_visit_sentences[visit_time] = []

            if visit_to_embed_static_variables == 'FIRST':
                if not first_visit_embedded:
                    d_visit_sentences[visit_time].extend(static_embeddings)
                    first_visit_embedded = True
            else:
                if visit_to_embed_static_variables == 'ALL':
                    d_visit_sentences[visit_time].extend(static_embeddings)

            d_visit_sentences[visit_time].extend(df_visit['Value'].to_list())
            d_visit_sentences[visit_time].extend(['SEP'])

        sample_dict['VisitSentences'] = d_visit_sentences

        return sample_dict


class OpGenerateRandomTrajectoryOfVisits(OpBase):
    def __call__(self, sample_dict) -> Any:

        d_visits_sentences = sample_dict['VisitSentences']
        n_visits = len(d_visits_sentences)
        # TODO add filtering patients with small number of visits - add to configuration
        if n_visits < 10:
            return sample_dict

        # get random first and last visit for generating random trajectory
        # we use n_visits- 1 to keep the last visit for outcome in classifier
        # TODO: "3" should be configured
        # TODO: for case with embedding od static info in first sentence only,
        #  add embeddings sentence here as we can cut out the first one
        start_visit = randrange(int((n_visits - 1) / 3))
        stop_visit = n_visits - randrange(int((n_visits - 1) / 3)) - 1
        keys = list(d_visits_sentences.keys())
        trajectory_keys = keys[start_visit:stop_visit]
        next_visit = keys[stop_visit]

        # Build trajectory
        trajectory_sentences = []
        for k in keys:
            trajectory_sentences.extend(d_visits_sentences[k])

        # TODO add CLS and PAD words before and after the trajectory
        sample_dict['Trajectory'] = trajectory_sentences
        sample_dict['NextVisit'] = d_visits_sentences[next_visit]

        return sample_dict


class OpMapToCategorical(OpBase):

    def __call__(self, sample_dict, percentiles: dict) -> Any:

        # convert continuous measurements to categorical ones based on defined percentiles

        # mapping static clinical characteristics (Age, Gender, ICU type, Height, etc)
        for k in sample_dict['StaticDetails'].keys():
            sample_dict['StaticDetails'][k] = k + '_' + \
                                              str(np.digitize(sample_dict['StaticDetails'][k], percentiles[k]))
        # mapping labs exams and clinical characteristics captured during patients' stay in ICU
        if not sample_dict['Visits'].empty:
            sample_dict['Visits']['Value'] = sample_dict['Visits']. \
                apply(
                lambda row: row['Parameter'] + '_' + str(np.digitize(row['Value'], percentiles[row['Parameter']])),
                axis=1)

        return sample_dict


class PhysioNetCinC:

    @staticmethod
    def _read_raw_data(raw_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # read patients info and lab exams
        df = pd.DataFrame(columns=["PatientId", "Time", "Parameter", "Value"])
        data_sub_sets = ["set-a", "set-b"]
        for s in data_sub_sets:
            csv_files = glob.glob(os.path.join(raw_data_path + '/' + s, "*.txt"))
            for f in csv_files[1:10]:  # reducing the list temporarily for debugging
                patient_id = os.path.splitext(os.path.basename(f))[0]
                df_file = pd.read_csv(f)
                df_file = df_file.drop(df_file[(df_file['Parameter'] == 'RecordID')].index).reset_index(drop=True)
                df_file["PatientId"] = patient_id
                df = df.append(df_file)
        df.reset_index(inplace=True, drop=True)
        patient_ids = np.unique(df['PatientId'].values)

        # read outcomes
        df_outcomes = pd.DataFrame(columns=["RecordID", "In-hospital_death"])
        outcomes = ['Outcomes-a.txt', 'Outcomes-b.txt']
        for o in outcomes:
            o_file = os.path.join(raw_data_path + '/' + o)
            df_outcomes = df_outcomes.append(pd.read_csv(o_file)[["RecordID", "In-hospital_death"]]).reset_index(
                drop=True)
        df_outcomes['RecordID'] = df_outcomes['RecordID'].astype(str)
        df_outcomes.rename(columns={'RecordID': 'PatientId'}, inplace=True)

        # synchronize with patients data
        df_outcomes = df_outcomes[df_outcomes['PatientId'].isin(patient_ids)]

        return df, df_outcomes

    @staticmethod
    def _drop_records_with_errors(df: pd.DataFrame) -> pd.DataFrame:
        # drop records with measurements below or equal to zero for tests with only positive values
        for v in VALID_TESTS_ABOVE_ZERO:
            df = df.drop(df[(df['Parameter'] == v) & (df['Value'] <= 0)].index).reset_index(drop=True)

        # drop gender values below zero
        df = df.drop(df[(df['Parameter'] == 'Gender') & (df['Value'] < 0)].index).reset_index(drop=True)

        # drop records with invalid Height and Weight values
        df = df.drop(df[(df['Parameter'] == 'Weight') & (df['Value'] < 20)].index).reset_index(drop=True)
        df = df.drop(df[(df['Parameter'] == 'Height') & (df['Value'] < 100)].index).reset_index(drop=True)
        return df

    @staticmethod
    def _drop_short_time_patients(df: pd.DataFrame, min_hours: int) -> pd.DataFrame:
        # remove patients that have the less than min_hours reported in ICU
        df_fixed = df.copy()
        count_dropped = 0
        for pat_id, df_pat_records in df.groupby('PatientId'):
            hours = df_pat_records['Time'].str.split(':', 1, True)[0].values
            if max(hours.astype(int)) < min_hours:
                df_fixed.drop(df_pat_records.index, inplace=True)
                count_dropped += 1
        print("Dropped " + str(count_dropped) + "short time patients")
        return df_fixed

    @staticmethod
    # TODO meanwhile didn't use this DATETIME - need to see if needed for the example
    def _convert_time_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
        # dummy date is added for converting time to date time format
        split = df['Time'].str.split(':', 1, True)
        df['Year'] = "2020"
        df['Month'] = "01"
        hour = split[0].astype(int)
        df['Day'] = np.where(hour >= 24, '02', '01')
        df['Hour'] = np.where(hour >= 24, hour - 24, hour)
        df['minute'] = split[1].astype(int)
        df['second'] = 0
        df["DateTime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "minute", "second"]])
        df.drop(columns=["Year", "Month", "Day", "Hour", "minute", "second"], inplace=True)

        return df

    @staticmethod
    def _generate_percentiles(dataset: DatasetDefault, num_percentiles: int,
                              categorical_max_num_of_values: int) -> dict:

        # TODO use debug mode and remove worker parameter when debugging is needed
        df = ExportDataset.export_to_dataframe(dataset, keys=['StaticDetails', 'Visits'], workers=1)

        # Extracting static and dynamic parts of the dataset
        df_static = pd.DataFrame(df['StaticDetails'].to_list())
        df_visits = pd.concat(df['Visits'].values)

        # generation dictionaries of values for static and dynamic variables of dataset patients
        d_static = df_static.to_dict('list')
        d_visits = dict.fromkeys(np.unique(df_visits[['Parameter']]), [])
        for k in d_visits.keys():
            d_visits[k] = df_visits[df_visits['Parameter'] == k]['Value']

        d_all_values = d_visits
        d_all_values.update(d_static)

        # calculate percentiles
        # for categorical parameters (Gender, etc) update percentiles according to categories
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
                # incrementing in 1 is needed for getting bin number corresponding to the variable value
                # e.g. gender values will remain the same
                unique_values = [x + 1 for x in unique_values]
                d_percentile[k] = sorted(unique_values)
                print("Categorical: " + k)
            else:
                d_percentile[k] = np.percentile(values, percentiles)

        return d_percentile

    @staticmethod
    def _load_and_process_df(raw_data_path: str, num_percentiles: int, min_hours_in_hospital: int) -> Tuple[
        pd.DataFrame, pd.DataFrame, dict, dict]:
        # if pickle available
        try:
            df_raw_data, df_outcomes, patient_ids = pickle.load(
                open(os.path.join(raw_data_path + '/' + 'raw_data.pkl'), "rb"))
        except:
            df_raw_data, df_outcomes = PhysioNetCinC._read_raw_data(raw_data_path)

            # drop records with invalid tests results
            df_raw_data = PhysioNetCinC._drop_records_with_errors(df_raw_data)

            # drop patients with less than MIN_HOURS hours in hospital
            df_raw_data = PhysioNetCinC._drop_short_time_patients(df_raw_data, min_hours_in_hospital)
            patient_ids = np.unique(df_raw_data['PatientId'].values)

            # fix time to datetime
            df_raw_data = PhysioNetCinC._convert_time_to_datetime(df_raw_data).reset_index()

            with open(os.path.join(raw_data_path + '/' + 'raw_data.pkl'), "wb") as f:
                pickle.dump([df_raw_data, df_outcomes, patient_ids], f)

        return df_raw_data, df_outcomes, patient_ids  # , dict_percentiles, dict_patient_time_events

    @staticmethod
    def _process_dynamic_pipeline(dict_percentiles, visit_to_embed_static_variables,
                                  static_variables_to_embed):
        return [
            (OpMapToCategorical(), dict(percentiles=dict_percentiles)),
            (OpConvertVisitToSentense(), dict(visit_to_embed_static_variables=visit_to_embed_static_variables, \
                                              static_variables_to_embed=static_variables_to_embed)),
            (OpGenerateRandomTrajectoryOfVisits(), dict())
        ]

    @staticmethod
    def dataset(
            raw_data_path: str,
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
            visit_to_embed_static_variables: str,
            static_variables_to_embed: Sequence[str],
    ) -> DatasetDefault:
        assert raw_data_path is not None

        df_records, df_outcomes, patient_ids = PhysioNetCinC._load_and_process_df(raw_data_path, num_percentiles,
                                                                                  min_hours_in_hospital)

        # first step of the pipline is to read data and add additional fatures (e.g. BMI)
        dynamic_pipeline_ops = [
            (OpReadDataframeCinC(df_records, outcomes=df_outcomes[['PatientId', 'In-hospital_death']],
                                 key_column='PatientId'), {}),
            (OpAddBMI(), dict())
        ]
        dynamic_pipeline = PipelineDefault("cinc_dynamic", dynamic_pipeline_ops)

        dataset_all = DatasetDefault(patient_ids, dynamic_pipeline)
        dataset_all.create()
        print("before balancing")
        folds = dataset_balanced_division_to_folds(
            dataset=dataset_all,
            output_split_filename=split_filename,
            keys_to_balance=['Target'],  # ["data.gt.probSevere"],
            nfolds=num_folds,
            seed=seed,
            reset_split=reset_split,
            workers=1
        )

        print("before dataset train")
        train_sample_ids = []
        for fold in train_folds:
            train_sample_ids += folds[fold]
        dataset_train = DatasetDefault(train_sample_ids, dynamic_pipeline)
        dataset_train.create()

        # calculate statistics of train set only and generate dictionary of percentiles for mapping
        # lab results to categorical for train, validation and test
        dict_percentiles = PhysioNetCinC._generate_percentiles(dataset_train, num_percentiles,
                                                               categorical_max_num_of_values)
        # TODO build vocabulary here based on defined percentiles + special words

        # update pypline with Op using calculated percentiles
        dynamic_pipeline_ops = dynamic_pipeline_ops + [
            *PhysioNetCinC._process_dynamic_pipeline(dict_percentiles, visit_to_embed_static_variables,
                                                     static_variables_to_embed)
        ]
        dynamic_pipeline = PipelineDefault("cinc_dynamic", dynamic_pipeline_ops)

        # update dynamic pipeline after calculating general statistics and percentiles based on train dataset
        dataset_train._dynamic_pipeline = dynamic_pipeline
        for f in train_sample_ids:
            x = dataset_train[f]

        print("before dataset val")
        validation_sample_ids = []
        for fold in validation_folds:
            validation_sample_ids += folds[fold]
        dataset_validation = DatasetDefault(validation_sample_ids, dynamic_pipeline)
        dataset_validation.create()
        for f in validation_sample_ids:
            x = dataset_validation[f]

        print("before dataset test")
        test_sample_ids = []
        for fold in test_folds:
            test_sample_ids += folds[fold]
        dataset_test = DatasetDefault(test_sample_ids, dynamic_pipeline)
        dataset_test.create()
        for f in test_sample_ids:
            x = dataset_test[f]

        return dataset_train, dataset_validation, dataset_test


if __name__ == "__main__":
    import os

    # from fuse.data.utils.export import ExportDataset

    ds_train, ds_valid, ds_test = PhysioNetCinC.dataset(SOURCE, 5, None, 1234, True, [0, 1, 2], [3], [4])
    # df = ExportDataset.export_to_dataframe(ds_train, ["activity.label"])
    # print(f"Train stat:\n {df['activity.label'].value_counts()}")
    # df = ExportDataset.export_to_dataframe(ds_valid, ["activity.label"])
    # print(f"Valid stat:\n {df['activity.label'].value_counts()}")
    # df = ExportDataset.export_to_dataframe(ds_test, ["activity.label"])
    # print(f"Test stat:\n {df['activity.label'].value_counts()}")
