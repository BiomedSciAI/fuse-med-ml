from typing import Sequence, Any
import re
import glob
import os

import numpy as np
import pandas as pd
from typing import Tuple

from fuse.data import DatasetDefault, PipelineDefault, OpBase
from fuse.data.ops.ops_read import OpReadDataframe
#from fuse.data.ops.ops_common import OpCond, OpSet
from fuse.data.utils.split import dataset_balanced_division_to_folds


SOURCE = r'C:/D_Drive/Projects/EHR_Transformer/PhysioNet/predicting-mortality-of-icu-patients-the-physionetcomputing-in-cardiology-challenge-2012-1.0.0/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0'

VALID_TESTS_ABOVE_ZERO = ['pH', 'Weight', 'Height', 'DiasABP', 'HR', 'NIMAP', 'MAP', 'NIDiasABP', 'NISysABP', 'PaCO2', 'PaO2', 'Temp', 'SaO2', 'RespRate', 'SysABP']

STATIC_FIELDS = ['Age','Gender','Height','ICUType','Weight']


class OpProcessTargetActivities(OpBase):
    def __call__(self, sample_dict) -> Any:

        sample_dict["activity.label"] = -1  # unknown

        return sample_dict


class OpProcessTargetActivities2(OpBase):
    def __call__(self, sample_dict) -> Any:

        sample_dict["activity.value"] = 1  # unknown

        return sample_dict

class PhysioNetCinC:

    @staticmethod
    def _read_raw_data(raw_data_path):
        df = pd.DataFrame(columns=["PatientId", "Time", "Parameter", "Value"])
        sub_sets = ["set-a", "set-b"]
        for s in sub_sets:
            csv_files = glob.glob(os.path.join(raw_data_path + '/' + s, "*.txt"))
            for f in csv_files[0:5]:  #reducung the list temporarely for debugging
                patient_id = os.path.splitext(os.path.basename(f))[0]
                df_file = pd.read_csv(f)
                df_file = df_file.drop(df_file[(df_file['Parameter'] == 'RecordID')].index).reset_index(drop=True)
                df_file["PatientId"] = patient_id
                df = df.append(df_file)

        df.reset_index(inplace=True,drop=True)
        return df

    @staticmethod
    def _drop_errors(df: pd.DataFrame) -> pd.DataFrame:
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
    def _generate_percentiles(df: pd.DataFrame, num_percentiles: int) -> dict:
        percentiles = range(0, 100 + int(100/num_percentiles), int(100/num_percentiles))

        list_params = np.unique(df[['Parameter']])

        # dictionary of values of each lab/vital test with percentiles
        d_values = dict.fromkeys(list_params, [])
        d_percentile = dict.fromkeys(list_params, [])

        for k in d_values.keys():
            d_values[k] = df[df['Parameter'] == k]['Value']
            d_percentile[k] = np.percentile(d_values[k], percentiles)

        return d_percentile

    @staticmethod
    def _convert_to_patients_df(df: pd.DataFrame) -> pd.DataFrame:
        # dict of patients

        statis_fields = ['Age', 'Height', 'Weight', 'Gender', 'ICUType']
        df_patients = pd.DataFrame(columns=['PatientId', 'BMI']+statis_fields)
        dict_patients_time_events = dict()
        idx = 0
        for pat_id, pat_records in df.groupby('PatientId'):
            df_patients.loc[idx, 'PatientId'] = pat_id
            df_static = pat_records[pat_records['Time'] == '00:00']
            for f in statis_fields:
                rec = df_static['Value'][(df_static['Parameter'] == f) & (df_static['Value'] >= 0)].reset_index(drop=True)
                if not rec.empty:
                    df_patients.loc[idx, f] = rec[0]


            if ~np.isnan(df_patients.loc[idx, 'Height']) & ~np.isnan(df_patients.loc[idx,'Weight']):
                df_patients.loc[idx, 'BMI'] = 10000 * df_patients.loc[idx, 'Weight'] / (df_patients.loc[idx, 'Height'] * df_patients.loc[idx, 'Height'])

            pat_records = pat_records.drop(pat_records[pat_records['Time'] == '00:00'].index).reset_index(drop=True)
            dict_patients_time_events[pat_id] = {time: tests.groupby('Parameter')['Value'].apply(list).to_dict()
                                                         for time, tests in pat_records[['DateTime', 'Parameter', 'Value']].groupby('DateTime')}
            idx = idx + 1

        return df_patients
        #
        #     dict_patients_df = {k: v.drop('PatientId', axis=1).reset_index(drop=True) for k, v in
        #                     df.groupby('PatientId')}
        #
        # dict_patients_nested = {
        #     k: {t: tt.groupby('Parameter')['Value'].apply(list).to_dict() for t, tt in f.groupby('Time')}
        #     for k, f in df.groupby('PatientId')}
        # df_patients.loc[idx, 'TimeEvents'] = {time: tests.groupby('Parameter')['Value'].apply(list).to_dict()
        #                                                   for time, tests in pat_records[['DateTime','Parameter','Value']].groupby('DateTime')}

    @staticmethod
    def _load_and_process_df(raw_data_path: str, num_percentiles: int) -> Tuple[pd.DataFrame,dict]:
        df = PhysioNetCinC._read_raw_data(raw_data_path)

        # drop records with invalid tests results
        df = PhysioNetCinC._drop_errors(df)

        # fix time to datetime
        df = PhysioNetCinC._convert_time_to_datetime(df)

        # define dictionary of percentiles
        percentiles_dict = PhysioNetCinC._generate_percentiles(df, num_percentiles)

        # reset index to reenumerate the samples
        df = df.reset_index()

        # build data frame of patients (one record per patient)
        df_patients = PhysioNetCinC._convert_to_patients_df(df)

        #add bmi


        return df_patients, percentiles_dict



    @staticmethod
    def _process_pipeline():
        return [
            (OpProcessTargetActivities(), dict()),

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
            num_percentiles: int
    ) -> DatasetDefault:
        assert raw_data_path is not None

        df = PhysioNetCinC._load_and_process_df(raw_data_path, num_percentiles)
        dynamic_pipeline = [
            (OpReadDataframe(df, key_column="PatientId"), {}),
            *PhysioNetCinC._process_pipeline()
        ]
        dynamic_pipeline = PipelineDefault("cinc", dynamic_pipeline)

        dataset_all = DatasetDefault(len(df), dynamic_pipeline)
        dataset_all.create()

        folds = dataset_balanced_division_to_folds(
            dataset=dataset_all,
            output_split_filename=split_filename,
            keys_to_balance=[],  # ["data.gt.probSevere"],
            nfolds=num_folds,
            seed=seed,
            reset_split=reset_split,
        )

        train_sample_ids = []
        for fold in train_folds:
            train_sample_ids += folds[fold]
        dataset_train = DatasetDefault(train_sample_ids, dynamic_pipeline)
        dataset_train.create()

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

        return dataset_train, dataset_validation, dataset_test


if __name__ == "__main__":
    import os
    #from fuse.data.utils.export import ExportDataset


    ds_train, ds_valid, ds_test = PhysioNetCinC.dataset(SOURCE, 5, None, 1234, True, [0, 1, 2], [3],[4])
    # df = ExportDataset.export_to_dataframe(ds_train, ["activity.label"])
    # print(f"Train stat:\n {df['activity.label'].value_counts()}")
    # df = ExportDataset.export_to_dataframe(ds_valid, ["activity.label"])
    # print(f"Valid stat:\n {df['activity.label'].value_counts()}")
    # df = ExportDataset.export_to_dataframe(ds_test, ["activity.label"])
    # print(f"Test stat:\n {df['activity.label'].value_counts()}")