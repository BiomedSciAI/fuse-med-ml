import pickle
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

        sample_dict["BMI"] = np.nan

        if ("Height" in sample_dict.keys()) & ("Weight" in sample_dict.keys()):
            height = sample_dict["Height"]
            weight = sample_dict["Weight"]
            if ~np.isnan(height) & ~np.isnan(weight):
                sample_dict["BMI"] = 10000 * weight / (height * height)

        print(sample_dict["BMI"])
        print(sample_dict['Visits'].shape)
        return sample_dict


# class OpCollectExamsStatistics(OpBase):
#     def __call__(self, sample_dict, percentiles: dict) -> Any:
#         percentiles[sample_dict['PatientId']] = sample_dict['Age']
#
#         return sample_dict


class OpMapToCategorical(OpBase):
    @staticmethod
    def _digitize(sample_dict, percentiles):
        #  sample_dict_mapped = np.deepcopy(sample_dict)
        #
        #  categorical_static = ['Gender', 'ICUType']
        #  #fix categorical in dictionary not in handling
        #  percentiles['Gender'] = [0,1]
        #  percentiles['MechVent'] = [0, 1]
        #
        #
        #  # mapping static clinical characteristics
        #  for k in sample_dict.keys():
        #      if k in categorical:
        #          sample_dict_mapped[k] = k + '_' + sample_dict[k]
        #      else: sample_dict_mapped[k] = np.digitize(sample_dict[k], percentiles[k])
        #
        #  df_visits = sample_dict['Visits']
        #
        #
        #     list_remaining = [x for x in list_params if x not in special_handling]
        #     for p in list_remaining:
        #         new_values = np.digitize(d_values[p], d_percentiles[p])
        #         new_deltas = np.digitize(d_delta[p], d_percentiles_deltas[p])
        #         mapped_values = [p + '_' + str(x) for x in new_values]
        #         mapped_deltas = [p + '_D_' + str(x) for x in new_deltas]
        #         df_mapped.loc[d_values[p].index, 'MappedValue'] = mapped_values
        #         df_mapped.loc[d_values[p].index, 'MappedDelta'] = mapped_deltas
        #
        #     if USE_DELTAS_FOR_MEASUREMETS:
        #         df_mapped['FinalMappedValue'] = np.where(df_mapped['FirstIndicator'], df_mapped['MappedValue'],
        #                                                  df_mapped['MappedDelta'])
        #     else:
        #         df_mapped['FinalMappedValue'] = df_mapped['MappedValue']
        #
        #     # remove records with patient ID
        #     df_mapped = df_mapped.drop(df_mapped[(df_mapped['Parameter'] == 'RecordID')].index).reset_index(drop=True)
        #
        #     with open(RESULTS + 'mapped_data.pkl', 'bw') as f:
        #         pickle.dump(df_mapped, f)
        # else:
        #     with open(RESULTS + 'mapped_data.pkl', 'rb') as f:
        #         df_mapped = pickle.load(f)

        return sample_dict

    def __call__(self, sample_dict, percentiles: dict) -> Any:
        print(percentiles['HR'])
        sample_dict = OpMapToCategorical._digitize(sample_dict, percentiles)
        # print(percentiles.keys())
        return sample_dict


class PhysioNetCinC:

    @staticmethod
    def _read_raw_data(raw_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # read patients info & tests
        df = pd.DataFrame(columns=["PatientId", "Time", "Parameter", "Value"])
        data_sub_sets = ["set-a", "set-b"]
        for s in data_sub_sets:
            csv_files = glob.glob(os.path.join(raw_data_path + '/' + s, "*.txt"))
            for f in csv_files[0:5]:  # reducung the list temporarely for debugging
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

        return df, df_outcomes, patient_ids

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
    def _generate_percentiles(dataset: DatasetDefault, num_percentiles: int) -> dict:

        # calculate statistics of train set only and generate dictionary of percentiles for mapping
        # lab results to categorical for train, validation and test
        df = ExportDataset.export_to_dataframe(dataset, keys=['StaticDetails', 'Visits'])

        df_static = pd.DataFrame(df['StaticDetails'].to_list())
        d_static = df_static.to_dict('list')

        # df_train_visits = ExportDataset.export_to_dataframe(dataset, keys=['Visits'])

        # combine data frames of all patients together and calculate statistics and percentiles
        # df = pd.concat(df_train_visits['Visits'].values)
        df_visits = pd.concat(df['Visits'].values)

        #list_params = np.unique(df_visits[['Parameter']])

        # dictionary of values of each lab/vital test with percentiles
        d_visits = dict.fromkeys(np.unique(df_visits[['Parameter']]), [])

        for k in d_visits.keys():
            d_visits[k] = df_visits[df_visits['Parameter'] == k]['Value']

        d_all_values = d_visits
        d_all_values.update(d_static)

        d_percentile = dict()
        percentiles = range(0, 100 + int(100 / num_percentiles), int(100 / num_percentiles))
        for k in d_all_values.keys():
            values = np.array(d_all_values[k])
            values = values[~np.isnan(values)]
            d_percentile[k] = np.percentile(values, percentiles)

        return d_percentile

    #
    # @staticmethod
    # def _convert_to_patients_df(df: pd.DataFrame, df_outcomes: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    #     # dict of patients
    #
    #     statis_fields = ['Age', 'Height', 'Weight', 'Gender', 'ICUType', 'In-hospital_death']
    #     df_patients = pd.DataFrame(columns=['PatientId', 'BMI']+statis_fields)
    #     dict_patients_time_events = dict()
    #     idx = 0
    #     for pat_id, pat_records in df.groupby('PatientId'):
    #         df_patients.loc[idx, 'PatientId'] = pat_id
    #         df_static = pat_records[pat_records['Time'] == '00:00']
    #         for f in statis_fields:
    #             rec = df_static['Value'][(df_static['Parameter'] == f) & (df_static['Value'] >= 0)].reset_index(drop=True)
    #             if not rec.empty:
    #                 df_patients.loc[idx, f] = rec[0]
    #
    #         # keep time events in dictionary    #
    #         pat_records = pat_records.drop(pat_records[pat_records['Time'] == '00:00'].index).reset_index(drop=True)
    #         dict_patients_time_events[pat_id] = {time: tests.groupby('Parameter')['Value'].apply(list).to_dict()
    #                                                      for time, tests in pat_records[['DateTime', 'Parameter', 'Value']].groupby('DateTime')}
    #
    #         # add outcome
    #         outcome = df_outcomes[df_outcomes['PatientId'] == pat_id]['In-hospital_death'].reset_index(drop=True)
    #         if not outcome.empty:
    #             df_patients.loc[idx, 'In-hospital_death'] = outcome[0]
    #
    #         idx = idx + 1
    #
    #     return df_patients, dict_patients_time_events
    #
    #     dict_patients_df = {k: v.drop('PatientId', axis=1).reset_index(drop=True) for k, v in
    #                     df.groupby('PatientId')}
    #
    # dict_patients_nested = {
    #     k: {t: tt.groupby('Parameter')['Value'].apply(list).to_dict() for t, tt in f.groupby('Time')}
    #     for k, f in df.groupby('PatientId')}
    # df_patients.loc[idx, 'TimeEvents'] = {time: tests.groupby('Parameter')['Value'].apply(list).to_dict()
    #                                                   for time, tests in pat_records[['DateTime','Parameter','Value']].groupby('DateTime')}
    # df.groupby('PatientId').apply(lambda x: x.set_index('DateTime').groupby('DateTime').apply( lambda y: y.to_numpy().tolist()).to_dict())

    @staticmethod
    def _load_and_process_df(raw_data_path: str, num_percentiles: int) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
        # if pickle avaialable
        try:
            df_raw_data, df_outcomes, patient_ids = pickle.load(
                open(os.path.join(raw_data_path + '/' + 'raw_data.pkl'), "rb"))
            # with open(os.path.join(raw_data_path + '/' + 'raw_data.pkl'), "rb") as f:
            #     return pickle.load(f)
        except:
            df_raw_data, df_outcomes, patient_ids = PhysioNetCinC._read_raw_data(raw_data_path)

            # drop records with invalid tests results
            df_raw_data = PhysioNetCinC._drop_errors(df_raw_data)

            # fix time to datetime
            df_raw_data = PhysioNetCinC._convert_time_to_datetime(df_raw_data).reset_index()

            # define dictionary of percentiles
            # dict_percentiles = PhysioNetCinC._generate_percentiles(df_raw_data, num_percentiles)

            # build data frame of patients (one record per patient)
            # df_patients, dict_patient_time_events = PhysioNetCinC._convert_to_patients_df(df_raw_data, df_outcomes)

            # df_raw_data = df_raw_data[['PatientId', 'DateTime', 'Parameter', 'Value']]
            with open(os.path.join(raw_data_path + '/' + 'raw_data.pkl'), "wb") as f:
                pickle.dump([df_raw_data, df_outcomes, patient_ids], f)

        return df_raw_data, df_outcomes, patient_ids  # , dict_percentiles, dict_patient_time_events

    # @staticmethod
    # def _process_static_pipeline(dict_percentiles):
    #     return [
    #         (OpAddBMI(), dict()),
    #
    #
    #     ]

    @staticmethod
    def _process_dynamic_pipeline(dict_percentiles):
        return [
            (OpAddBMI(), dict()),
            (OpMapToCategorical(), dict(percentiles=dict_percentiles))
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

        df_records, df_outcomes, patient_ids = PhysioNetCinC._load_and_process_df(raw_data_path, num_percentiles)

        # TODO: could we do data frame read w/o pipeline, verify with Moshico for rebuilding dynamic pipeline?
        dynamic_pipeline_ops = [
            (OpReadDataframeCinC(df_records, outcomes=df_outcomes[['PatientId', 'In-hospital_death']],
                                 key_column='PatientId'), {}),
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

        # df_train_static = ExportDataset.export_to_dataframe(dataset_train, keys=['Visits'])
        # calculate statistics of train set only and generate dictionary of percentiles for mapping
        # lab results to categorical for train, validation and test
        # df_train_visits = ExportDataset.export_to_dataframe(dataset_train, keys=['Visits'])

        # combine data frames of all patients together and calculate statistics and percentiles
        # df_train_visits_combined = df = pd.concat(df_train_visits['Visits'].values)
        # dict_percentiles = PhysioNetCinC._generate_percentiles(pd.concat(df_train_visits['Visits'].values),
        #                                                      num_percentiles)
        dict_percentiles = PhysioNetCinC._generate_percentiles(dataset_train, num_percentiles)

        # update pypline with Op using calculated percentiles
        dynamic_pipeline_ops = dynamic_pipeline_ops + [
            *PhysioNetCinC._process_dynamic_pipeline(dict_percentiles)
        ]
        dynamic_pipeline = PipelineDefault("cinc_dynamic", dynamic_pipeline_ops)
        dataset_train._dynamic_pipeline = dynamic_pipeline
        for f in train_sample_ids:
            x = dataset_train[0]

        print("before dataset val")
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

    # from fuse.data.utils.export import ExportDataset

    ds_train, ds_valid, ds_test = PhysioNetCinC.dataset(SOURCE, 5, None, 1234, True, [0, 1, 2], [3], [4])
    # df = ExportDataset.export_to_dataframe(ds_train, ["activity.label"])
    # print(f"Train stat:\n {df['activity.label'].value_counts()}")
    # df = ExportDataset.export_to_dataframe(ds_valid, ["activity.label"])
    # print(f"Valid stat:\n {df['activity.label'].value_counts()}")
    # df = ExportDataset.export_to_dataframe(ds_test, ["activity.label"])
    # print(f"Test stat:\n {df['activity.label'].value_counts()}")
