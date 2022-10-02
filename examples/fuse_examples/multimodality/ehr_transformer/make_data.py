import socket
import os
import pandas as pd
import pickle
from tqdm import tqdm
from ehrtransformers.model.utils import WordVocab
from ehrtransformers.configs.config import get_vocab_path
import numpy as np
from ehrtransformers.configs.config import get_config
from datetime import datetime

def validate_naming_conventions(naming_conventions):
    assert 'patient_id_key' in naming_conventions
    assert 'index_date_key' in naming_conventions
    assert 'svc_date_key' in naming_conventions #not strictly necessary - we can do without this column, if admission date column (adm_date_key) exists
    assert 'adm_date_key' in naming_conventions
    assert 'index_date_key' in naming_conventions
    assert 'age_key' in naming_conventions
    # assert 'age_month_key'  in naming_conventions
    assert 'date_birth_key' in naming_conventions
    assert 'gender_key' in naming_conventions
    # assert 'dxver_key'  in naming_conventions
    assert 'outcome_key' in naming_conventions

    assert isinstance(naming_conventions['patient_id_key'], str)
    assert isinstance(naming_conventions['index_date_key'], str)
    assert isinstance(naming_conventions['svc_date_key'], str)
    assert isinstance(naming_conventions['adm_date_key'], str)
    assert isinstance(naming_conventions['age_key'], str)
    # assert isinstance(naming_conventions['age_month_key'], str)
    assert isinstance(naming_conventions['date_birth_key'], str)
    assert isinstance(naming_conventions['gender_key'], str)
    # assert isinstance(naming_conventions['dxver_key'], str)
    assert isinstance(naming_conventions['outcome_key'], str)


class TimeFilter():
    date_column_name: str = None  # name of the column containing (visit) dates to be filtered
    ref_date_column_name = None  # name of a column containing a date to be used as a reference for filtering (e.g. index date)
    start_days_before = None  # visits ealier than start_days_before before the reference date are filtered out. May be negative (in which case treated as days after ref. date)
    stop_days_before = None  # visits later than stop_days_before before reference date are filtered out. May be negative (in which case treated as days after ref. date)

    def __init__(self, start_days_before=None, stop_days_before=None, date_column_name=None, ref_date_column_name=None):
        """

        :param start_days_before: visits ealier than start_days_before before the reference date are filtered out. May
        be negative (in which case treated as days after ref. date). If None - all visits are taken.
        :param stop_days_before: visits later than stop_days_before before reference date are filtered out. May be
        negative (in which case treated as days after ref. date). If None - all visits are taken.
        :param date_column_name: name of the column containing (visit) dates to be filtered
        :param ref_date_column_name: name of a column containing a date to be used as a reference for filtering (e.g. index date)
        """
        self.date_column_name = date_column_name
        self.start_days_before = start_days_before
        self.stop_days_before = stop_days_before
        self.ref_date_column_name = ref_date_column_name
        self.date_format_string = '%Y-%m-%d' # TODO: This may not be the format in the other input df's - be careful when switching to other data sources

    def apply_filter(self, df):
        """
        Filters a dataframe of visits so that only visits whose date (self.date_column_name) is between ref_date_column_name-start_days_before and ref_date_column_name-stop_days_before are kept
        :param df: input dataframe
        :return: filtered dataframe
        """
        #Since, for some reason, some systems fail to compute datetime differences, we first translate dates into integers
        if self.stop_days_before is not None:
            if isinstance(df.iloc[0][self.ref_date_column_name], str):
                tmp = df[self.ref_date_column_name].apply(lambda x: datetime.strptime(x, self.date_format_string).date())
                df['days_before_index'] = tmp.apply(lambda x: x.year*365+x.month*31+x.day*1 if(pd.notnull(x)) else x) - df[self.date_column_name].apply(lambda x: x.year*365+x.month*31+x.day*1 if(pd.notnull(x)) else x)
            else:
                df['days_before_index'] = df[self.ref_date_column_name].apply(lambda x: x.year*365+x.month*31+x.day*1 if (pd.notnull(x)) else x) - df[self.date_column_name].apply(lambda x: x.year*365+x.month*31+x.day*1 if(pd.notnull(x)) else x)
            df = df[(df[self.ref_date_column_name].isna()) | (df['days_before_index'] > self.stop_days_before)]
        if self.start_days_before is not None:
            df['days_before_index'] = df[self.ref_date_column_name].apply(lambda x: x.year*365+x.month*31+x.day*1 if(pd.notnull(x)) else x) - df[self.date_column_name].apply(lambda x: x.year*365+x.month*31+x.day*1 if(pd.notnull(x)) else x)
            df = df[(df[self.ref_date_column_name].isna()) | (df['days_before_index'] < self.start_days_before)]
        return df

        # if self.stop_days_before is not None:
        #     if isinstance(df.iloc[0][self.ref_date_column_name], str):
        #         tmp = df[self.ref_date_column_name].apply(lambda x: datetime.strptime(x, self.date_format_string).date())
        #         df['days_before_index'] = tmp.apply(lambda x: x.year * 365 + x.month * 31 + x.day * 1) - df[
        #             self.date_column_name].apply(lambda x: x.year * 365 + x.month * 31 + x.day * 1)
        #     else:
        #         df['days_before_index'] = df[self.ref_date_column_name] - df[self.date_column_name]
        #     df = df[(df[self.ref_date_column_name].isna()) | (df['days_before_index'].dt.days > self.stop_days_before)]
        # if self.start_days_before is not None:
        #     df['days_before_index'] = df[self.ref_date_column_name] - df[self.date_column_name]
        #     df = df[(df[self.ref_date_column_name].isna()) | (df['days_before_index'].dt.days < self.start_days_before)]
        # return df


class SplitFilter():
    split_method = None
    tr_val_tst_ratios = None
    tr_val_tst_column = None
    split_key = None  # names of the split sets (how they appear in the split DF and how they're seen outside)
    split_dict = None
    split_level_key = None  # name of the column according to which to group the split. All entries with the same value in this column will end up in the same group after the split (e.g. patient ID)

    def __init__(self, split_method, tr_val_tst_names=['train', 'validation', 'test'], tr_val_tst_ratios=None,
                 tr_val_tst_column=None, split_dict=None, split_level_key=None):
        """
        :param split_method: 'in_data', 'new', 'split_dict': describes how to split the data - 'in_data': to use 'SPLIT' and 'FOLD' fields found in the input df (df_dict),
                'new': create a new split based on tr_val_tst_ratios, or 'split_dict': use dataframes (with 'SPLIT' and 'FOLD' fields per patient) found in split_dict under keys that correspond to df_dict
        :param tr_val_tst_ratios: ratios of train_val_test sets. A list with first value corresponding to train, then
        val, then test. There may be fewer/more values - the output will change accordingly. Must be set if split_method is 'new'
        :param split_dict: if not None, contains a dataframe with columns for patient ID, tr_val_tst_column (with one
        of tr_val_tst_db_names for each patient) and 'FOLD' (0-n_folds)
        :param tr_val_tst_names = how train, val and test sets are marked in data or in split_dict. Must be set if split_method is 'in_data' or 'split_dict'. Suggested:['train', 'validation', 'test']
        :param tr_val_tst_column: a column in split_dict or in the dataframe where the split is defined. Must be set if split_method is 'in_data' or 'split_dict'


        """
        self.split_method = split_method
        self.tr_val_tst_ratios = tr_val_tst_ratios
        self.split_key = tr_val_tst_column
        # self.split_key=
        self.tr_val_tst_names = tr_val_tst_names
        self.split_dict = split_dict
        self.split_level_key = split_level_key

        assert tr_val_tst_names != None
        if self.split_method == 'new':
            assert tr_val_tst_ratios != None
        if self.split_method == 'in_data':
            assert tr_val_tst_column != None
        if self.split_method == 'split_dict':
            assert tr_val_tst_column != None
            assert split_dict != None

    def apply_split(self, df):
        """
        Splits input DF and returns a dictionary of {split_name: split_df} for all split_names in self.tr_val_tst_names.
        NOTE: tr_val_tst_names need to be uniform across all ingested datasets, for now
        :param df:
        :param split_level_key: a column in df according to which to group the split. All lines with the same value in this column will end up in the same group after the split (e.g. patient ID)
        :return:
        """
        out_dfs = {}
        if self.split_method == 'new':  # use tr_val_tst ratios to split the df:
            # Split df into train, val, test:
            df[self.split_key] = np.random.choice(a=self.tr_val_tst_names, size=df.shape[0], replace=True,
                                                  p=self.tr_val_tst_ratios)
            # the above assigned random choice to every visit. Now we need to pick one per patient:
            grouped = df.groupby(self.split_level_key)
            df[self.split_key] = grouped[self.split_key].transform('first')
        elif self.split_method == 'split_dict':
            # TODO: use split_dict to get train/validation/test results
            df = df.merge(self.split_dict, how='inner', on=self.split_level_key, suffixes=('', '_right'))
        elif self.split_method == 'in_data':
            # do nothing here because there already is a column called self.split_key
            a = 1
        for ind_tr_val_tst, tr_val_tst_name in enumerate(self.tr_val_tst_names):
            df_tmp = df[df[self.split_key] == tr_val_tst_name]
            out_dfs[tr_val_tst_name] = df_tmp

        # TODO: Remove the columns added to the df
        return out_dfs


class DataContainer():
    output_naming_conventions = None  # Uniform names of relevant stored columns
    tr_val_tst_names = ['train', 'validation', 'test']
    corpus = None #corpus of all input symbols, to be used in a vocabulary builder later
    inddate_filter = {
        'train': None,
        'validation': None,
        'test': None
    }

    def __init__(self, output_naming_conventions):
        """

        :param output_naming_conventions: A dictionary describing the names of columns of the ingested dataframes (used, in part, to translate them from the input DB):
            Columns to be taken "as is" from the input DB, with only name changes:
                'patient_id_key' (e.g. 'ENROLID'): (string) Unique patient ID
                'index_date_key' (e.g. 'INDDATE'): (datetime) Index date of the disease - first diagnosis or first treatment (earlier of the two)
                'svc_date_key' (e.g. 'SVCDATE'): (datetime) Date of service (there may be several services per admission)
                'adm_date_key' (e.g. 'ADMDATE'): (datetime) Date of admission to a hospital (in case of a single day visit, e.g. for outpatients, this is the same as service date)
                'age_key' (e.g. 'AGE'): (int) Age in years
                'age_month_key' (e.g. 'AGE_MON'): (int)(optional) age in months. If doesn't exist, will be recreated using service/admission date and year of birth.
                'date_birth_key' (e.g. 'DOBYR'): Year of birth
                'gender_key' (e.g. 'SEX'): (string) Patient sex at birth(?)
                'dxver_key' (e.g. 'DXVER'): (string) Diagnosis code version (e.g. ICD9 or ICD10)
                'outcome_key' (e.g. 'PD'): (int 0 or 1) Indicator of whether or not the patient is part of the sick cohort

                'event_type_key': 'EVENT_TYPE', # (string)(optional, if additional GT table is used) general groups of events (as opposed to specific diagnostic codes) - these will most likely be the ones we'll try to predict
                'event_date_key': 'EVENT_DATE', # (string)(optional, if additional GT table is used) date of a given event
                'event_code_key': 'CODE',       # (datetime) specific event diagnostic code
                'event_code_type_key': 'CODE_TYPE' # diagnostic code convention (ICD-9, ICD-10, etc.)
            New columns:
                'date_key' (e.g. 'ADMDATE'): (datetime) Date of the visit (either admission date or service date, usually)

        """
        self.output_naming_conventions = output_naming_conventions

    def output_vocab(self, corpus_path, vocab_path):
        if self.corpus is not None:
            if not os.path.exists(os.path.dirname(corpus_path)):
                os.makedirs(os.path.dirname(corpus_path))

            with open(corpus_path, 'w') as f:
                for item in self.corpus:
                    line = "{} ".format(item)
                    f.write(line)
            with open(corpus_path, "r") as f:
                vocab = WordVocab(f, max_size=None, min_freq=1)

            vocab_out = {'token2idx': vocab.get_stoi(), 'idx2token': vocab.get_itos()}
            with open(vocab_path + '.pkl', 'bw') as f:
                pickle.dump(vocab_out, f)

class GTContainer(DataContainer):
    """
    Contains event GT information
    """
    a = 1
    main_df = None
    gt_field_names = []
    modes = None
    target_key = None #name of a column to store the resulting event GT values
    # corpus = [] #event corpus (containing all event names/symbols) to be later used in building an event vocabulary

    # noinspection PyShadowingNames
    def __init__(self, output_naming_conventions: dict, modes, target_key: str):
        """
        :type output_naming_conventions: dict
        :param output_naming_conventions: contains output field naming conventions:
        {
            age_key: age in years
            age_month_key: age in months
            date_key: ADMDATE
            diagnosis_vec_key: DX
            gender_key: SEX
            label_key: label
            outcome_key: PD
            patient_id_key: ENROLID
            separator_str: SEP
            healthy_val: 0
            sick_val: 1
        }

        output_db_naming_conventions['patient_id_key'] (e.g. 'ENROLID'): Unique patient ID (string)
        output_db_naming_conventions['index_date_key'] (e.g. 'INDDATE'): Index date of the disease (datetime) - first diagnosis or first treatment (earlier of the two)
        output_db_naming_conventions['svc_date_key'] (e.g. 'SVCDATE'): Date of service (datetime) (there may be several services per admission)
        output_db_naming_conventions['adm_date_key'] (e.g. 'ADMDATE'): Date of admission to a hospital (datetime) (in case of a single day visit, e.g. for outpatients, this is the same as service date)
        output_db_naming_conventions['age_key'] (e.g. 'AGE'): Age in years (int)
        output_db_naming_conventions['age_month_key'] (e.g. 'AGE_MON'): (int)(optional) age in months. If doesn't exist, will be recreated using service/admission date and year of birth.
        output_db_naming_conventions['date_birth_key'] (e.g. 'DOBYR'): Year of birth
        output_db_naming_conventions['gender_key'] (e.g. 'SEX'): (string) Patient sex at birth(?)
        output_db_naming_conventions['dxver_key'] (e.g. 'DXVER'): (string) Diagnosis code version (e.g. ICD9 or ICD10)
        output_db_naming_conventions['outcome_key'] (e.g. 'PD'): (int 0 or 1) Indicator of whether or not the patient is part of the sick cohort

        :param modes: gt addition mode. A string or a dictionary.
            If modes is a dictionary, it contains key:value pairs of gt_type:gt_time_ref, where gt_type is the name of a GT column,
            and gt_time_ref is another dictionary that dictates how the gt is predicted.
            Possible gt_time_ref are:
                'future_timeframe':n_days - an event is included in the GT if it happens within n_days after the current visit
                'any_time':None NOT IMPLEMENTED - an event is included in the GT if it happens at any time for this patient
            If modes is a string, it determines a given mode for all GT event types. Implemented options:
                'within_90_days': equivalent to gt_time_ref of {'future_timeframe':90} for all event types
                'any_time' : NOT IMPLEMENTED equivalent to gt_time_ref of {'any_time':None} for all event types
        """
        super().__init__(output_naming_conventions=output_naming_conventions)
        self.modes = modes
        self.corpus = []#list(modes.keys())
        self.target_key = target_key
        self.gt_field_names = []

    def add_event_df(self, df, event_names: list = None, event_naming_conventions: dict=None, same_visit_date_diff: int=1):
        """
        Adds groundtruth information from df to current event container. The input df should have a date column and an event type column. Input events are
        expected to be in form three columns: patient:event:date, with a line for every event.
        Events are stored as patient:certain_event:[list of all event occurrences], with a separate column for each event type
        TODO: test multiple df additions (so far - single one works, more than one - untested
        :param df:
        :param event_names: if not None, contains a list of event types to include in the GT. If None - all events in the df are included (TODO: the last part is not implemented)
        :param event_naming_conventions: dictionary containing the meanings of df column names. If None - self.output_naming_conventions are used. The following fields are expected:
        {
            'patient_id_key': 'ENROLID', # unique user id
            'event_type_key': 'EVENT_TYPE', # general groups of events (as opposed to specific diagnostic codes) - these will most likely be the ones we'll try to predict
            'event_date_key': 'EVENT_DATE', # date of a given event
            'event_code_key': 'CODE',       # specific event diagnostic code
            'event_code_type_key': 'CODE_TYPE' # (string) category of event
        }
        :param modes:
        :return:
        """
        in_df = df.copy(deep=True)  # to avoid changing the original dataframe
        if event_names == None:
            event_names = list(set(df[event_naming_conventions['event_type_key']]))

        if None in event_names:
            event_names = [e for e in event_names if e is not None]

        # create a column for each event that contains event dates
        for ev_name in event_names:
            in_df[ev_name] = in_df.apply(lambda x: x[event_naming_conventions['event_date_key']] if x[event_naming_conventions['event_type_key']] == ev_name else None,
                                         axis=1)
            self.gt_field_names.append(ev_name)
        # self.gt_field_names.extend(list(set(self.gt_field_names)))
        self.gt_field_names = list(set(self.gt_field_names))
        self.corpus.extend(self.gt_field_names)

        aggr_dict = {}
        for ev_name in event_names:
            aggr_dict[ev_name] = lambda x: [l for l in x if pd.notna(
                l)]  # TODO: check if we get double entries for a given event (resulting in date duplicates). This is improbable, and if it happens it won't cause a problem except for somewhat slowing overall calculations.
        #This results in a single row per patient, with a column per event type that contains a list of all dates when the event occurs
        #for the patient
        in_df = in_df.groupby([event_naming_conventions['patient_id_key']]).agg(aggr_dict).reset_index()

        data_column_renames = {
            event_naming_conventions['patient_id_key']: self.output_naming_conventions['patient_id_key'],
        }

        in_df.rename(columns=data_column_renames, inplace=True)

        if self.main_df == None:
            self.main_df = in_df
        else:
            #TODO: this won't work if we have the same event in both left and right (the right event will have a separate column). We need to merge the date lists for the events.
            self.main_df = self.main_df.merge(in_df, how='outer', on=self.output_naming_conventions['patient_id_key'],
                                              suffixes=('', '_right'))

        a = 1

    def add_gt_values_to_external_df(self, ext_df):
        """
        adds gt (from gt container) values to ext_df (data container) - a dataframe that concurs with self.output_naming_conventions
        :param ext_df: every line is a history of visits of a patient up to a certain date (of the last visit in the sequence)

        :return ext_df: Combined dataframe containing visit information ang GT
        :return all_event_types: all events that occurred in the DF
        """
        modes = self.modes
        if self.main_df is None:
            return ext_df, None
        if isinstance(modes, str):
            if modes == 'within_90_days':
                gt_time_ref = {'future_timeframe': 90}
            elif modes == 'within_60_days':
                gt_time_ref = {'future_timeframe': 60}
            else:
                raise Exception('Mode type {} not implemented'.format(modes))
            modes = {gt_ev_name: gt_time_ref for gt_ev_name in self.gt_field_names}
        elif isinstance(modes, dict):
            if 'all' in modes:
                gt_time_ref = modes['all']
                modes = {gt_ev_name: gt_time_ref for gt_ev_name in self.gt_field_names}

        #add a column per event that contains a list of all dates for this event for a patient, for every visit trajectory of the patient
        ext_df = ext_df.merge(self.main_df, how='left', on=self.output_naming_conventions['patient_id_key'],
                              suffixes=('', '_right'))

        for ev_name in modes:
            ext_df[ev_name + '_date'] = ext_df[ev_name]  # back up the event dates for debugging
            # compute time deltas from current admission date to every future event date:
            ext_df[ev_name] = ext_df.apply(
                lambda x: [(ev_date - x[self.output_naming_conventions['date_key']]).days for ev_date in x[ev_name] if
                           (ev_date - x[self.output_naming_conventions['date_key']]).days > 0] if isinstance(x[ev_name],
                                                                                                             list) else [],
                axis=1)
            # keep only the number of days to the nearest future event for every visit:
            ext_df[ev_name] = ext_df[ev_name].apply(lambda x: min(x) if len(x) > 0 else np.nan)

        for ev_name in modes:
            ev_time_ref = modes[ev_name]
            if 'future_timeframe' in ev_time_ref:
                time_delta = ev_time_ref['future_timeframe']
                ext_df[ev_name] = ext_df[ev_name].apply(lambda x: ev_name if (x < time_delta) else np.nan)
        all_gt_columns = list(modes.keys())

        # Collect values from all_gt_columns into a list, ignoring NaN's, and put the resulting list into self.output_naming_conventions['event_key']
        # ext_df[self.output_naming_conventions['event_key']] = ext_df[all_gt_columns].values.tolist() #this does not ignore NaN's
        ext_df[self.target_key] = ext_df[all_gt_columns].apply(
            lambda x: x.dropna().tolist(), axis=1)  # This collects into list, and ignores NaN's.

        date_columns = [evt+'_date' for evt in all_gt_columns]
        ext_df.drop(columns=all_gt_columns+date_columns, inplace=True)


        return ext_df, all_gt_columns


class InputDataContainer(DataContainer):
    # time_filters = None
    main_df = None
    max_age = 110
    # corpus = [] #corpus of all input symbols, to be used in a vocabulary builder later

    def __init__(self, output_naming_conventions=None, init_df=None):
        """

        :param init_df: initial dataframe that contains ingested data
        :param output_naming_conventions: A dictionary describing the columns of the ingested dataframe (used, in part, to translate them from the input DB):
            Columns to be taken "as is" from the input DB:
                output_db_naming_conventions['patient_id_key'] (e.g. 'ENROLID'): (string) Unique patient ID
                output_db_naming_conventions['index_date_key'] (e.g. 'INDDATE'): (datetime)Index date of the disease  - first diagnosis or first treatment (earlier of the two)
                output_db_naming_conventions['svc_date_key'] (e.g. 'SVCDATE'): (datetime)Date of service  (there may be several services per admission)
                output_db_naming_conventions['adm_date_key'] (e.g. 'ADMDATE'): (datetime)Date of admission to a hospital  (in case of a single day visit, e.g. for outpatients, this is the same as service date)
                output_db_naming_conventions['age_key'] (e.g. 'AGE'): Age in years (int)
                output_db_naming_conventions['age_month_key'] (e.g. 'AGE_MON'): (int)(optional) age in months. If it doesn't exist, will be recreated using service/admission date and year of birth.
                output_db_naming_conventions['date_birth_key'] (e.g. 'DOBYR'): Year of birth
                output_db_naming_conventions['gender_key'] (e.g. 'SEX'): (string) Patient sex at birth(?)
                output_db_naming_conventions['dxver_key'] (e.g. 'DXVER'): (string) Diagnosis code version (e.g. ICD9 or ICD10)
                output_db_naming_conventions['outcome_key'] (e.g. 'PD'): (int 0 or 1) Indicator of whether or not the patient is part of the sick cohort
            New columns:
                'date_key': (datetime) contains a date of the visit (usually same as admission date, but other choices are possible)
        """
        super().__init__(output_naming_conventions)
        self.main_df = init_df
        # self.time_filters = time_filters
        a = 1


    def ingest_df(self, input_dfs, input_columns, output_naming_conventions, task=None, fields_to_keep=None, same_visit_date_diff=1):
        """
        Adds a dataframe with data read from a DB to the data container. A list of diagnoses (or other relevant input fields as defined by diag_codes_outp)
         ending with output_naming_conventions['separator_str'] is generated for each visit.
         The dataframe needs to have the following columns, as defined in self.output_naming_conventions:
        patient_id_key = 'ENROLID': unique patient ID
        index_date_key = 'INDDATE'
        svc_date_key = 'SVCDATE'
        adm_date_key = 'ADMDATE'
        age_key = 'AGE'
        age_month_key = 'AGE_MON'
        date_birth_key = 'DOBYR'
        gender_key = 'SEX'
        dxver_key = 'DXVER'

        :param input_dfs: list of dataframes of all patient visit entries to be ingested (can be a single df)
        :param input_columns: a list of columns in the df that contain codes relevant for inputs
        :param task: name of the task (e.g. 'breast_cancer')
        :param fields_to_keep: a dictionary of {'column name' : 'aggr function name'} so that for every column name in the DB we want to keep,
            we'll have a function that will be used to aggregate the values of the column over a single visit of a given patient
        :param same_visit_date_diff: if a number of days between visit is less than this, they're merged into a single visit
        :return:
        """
        if not isinstance(input_dfs, list):
            input_dfs = [input_dfs]

        patient_id_key = self.output_naming_conventions['patient_id_key']  # "ENROLID"
        # ignore_columns = ['SEQNUM', patient_id_key, "REGION", "MDC", "INDDATE", "COHORT"]
        ignore_columns = ['SEQNUM', "REGION", "MDC", "COHORT", "MSA", "PROCTYP", "AGEGRP", "RX"]
        index_date_key = self.output_naming_conventions['index_date_key']  # 'INDDATE'
        svc_date_key = self.output_naming_conventions['svc_date_key']  # 'SVCDATE'
        adm_date_key = self.output_naming_conventions['adm_date_key']  # 'ADMDATE'
        # date_key = svc_date_key
        diag_codes_outp = input_columns   #diag code format

        age_key = self.output_naming_conventions['age_key']  # "AGE"
        age_month_key = self.output_naming_conventions['age_month_key']  # 'AGE_MON'
        date_birth_key = self.output_naming_conventions['date_birth_key']  # "DOBYR"
        sex_key = self.output_naming_conventions['gender_key']  # "SEX"
        dxver_key = self.output_naming_conventions['dxver_key']  # "DXVER"
        # outcome_key = self.output_naming_conventions['outcome_key'] #"TYPE" #unused - replaced by ext_gt and fields_to_keep

        split_key = output_naming_conventions['split_key']
        fold_key = output_naming_conventions['fold_key']

        diagnosis_col_new = output_naming_conventions['diagnosis_vec_key']
        end_of_sentence_sep = output_naming_conventions['separator_str']

        for df in tqdm(input_dfs):
            # In inpatient admissions table, an admission is usually several days, with a different service date for each, but
            # all with the same admission date. Therefore, if admission date is present, we use it to derive a single visit
            if adm_date_key in df:
                date_key = adm_date_key
            else:
                date_key = svc_date_key

            # In inpatients admissions there exist 15 diagnosis codes, and in outpatients - only 4:
            # The above comment is irrelevant because we do not look at inpatient admissions but at inpatient services, where there are only 4 diag codes

            # We're only dealing with inpatient/outpatient services, so we'll take diag_codes_outp. TODO: refactor this
            diag_codes = diag_codes_outp

            #If procedures are part of the input, their codes need to change, since in some protocols there may be
            # shared codes between diagnoses and procedures. It's done by adding a 'pr_' prefix to the values of all
            # columns that contain 'proc' in their name
            # TODO: make this more generic by reading procedure columns from a config, rather than trying to infer procedure column names
            for c in diag_codes:
                if 'proc' in c.lower():
                    df[c] = df[c].apply(lambda x: None if x is None else 'pr_'+str(x))
                else:
                    df[c] = df[c].apply(lambda x: None if x is None else str(x))


            df = df.drop(axis=0, columns=ignore_columns, errors='ignore')
            df[diagnosis_col_new] = df[diag_codes].values.tolist()

            #remove old diag_codes columns (keep diagnosis_col_new, if it's in diag_codes)
            diag_codes = set(diag_codes)
            if diagnosis_col_new in diag_codes:
                diag_codes.remove(diagnosis_col_new)
            if len(diag_codes)>0:
                df = df.drop(axis=0, columns=list(diag_codes), errors='ignore')

            df[diagnosis_col_new] = df[diagnosis_col_new].apply(lambda x: list(filter(None, list(set(x)))))
            # we now have a table with a list of diagnoses per visit of a patient
            #

            df = df.explode(diagnosis_col_new)
            df.drop_duplicates(inplace=True)

            # df = df.dropna() #This is commented out on purpose. We must be extra careful removing Nones from the
            # dataframe, because some columns may contain legitimate None values. Right now, I'll remove None's only from
            # the input columns as they are defined in diag_codes
            df = df[df[diagnosis_col_new].notna()] #TODO: check if all these dropna's are legit (maybe export the dropna column definitions to the config)
            # we now have a table where each diagnosis has its own line

            #Remove rows with undefined age or admission date or sex
            df.dropna(axis='rows', subset=[date_key, date_birth_key, sex_key], inplace=True)
            try: #(in case date_key contains Timestamp and not DateTime.date)
                df[date_key] = df[date_key].apply(lambda x: x.date())
            except:
                a=1 #do nothing

            if age_month_key not in df:
                df[age_month_key] = df.apply(
                    lambda x: (x[date_key].year - int(x[date_birth_key])) * 12 + x[date_key].month,
                    axis=1)  # TODO: This takes a LONG time to calculate. See if it can be made more efficient
                # This is not exactly age in months (for patients not born on 01/01/birth_year), but since we mainly
                # need age as a frame of reference, and have no exact birthdates, this will do.
            if age_key not in df:
                df[age_key] = df.apply(
                    lambda x: (x[date_key].year - x[date_birth_key]),
                    axis=1)


            all_dx_codes = list(df[diagnosis_col_new])
            if self.corpus is not None:
                self.corpus.extend(all_dx_codes)
            else:
                self.corpus = all_dx_codes
            self.max_age = max(list(df[age_key]))


            # df.groupby([patient_id_key]).apply(lambda x: x[date_key].diff())

            # Which columns to keep, and how to aggregate them by patient and visit date
            agg_dict = {diagnosis_col_new: lambda x: (
                    list(set([l for l in x if pd.notna(l)])) + [end_of_sentence_sep]),
                        age_key: lambda x: str(max(x)),
                        age_month_key: lambda x: str(max(x)),
                        date_birth_key: lambda x: max(x),
                        sex_key: lambda x: str(max(x)), #x.iloc[0],  # max(x),
                        # dxver_key: lambda x: x.iloc[0],  # max(x),
                        index_date_key: lambda x: x.iloc[0],  #
                        date_key: lambda x: x.iloc[0],
                        }
            # additional columns that we'll want to keep/process (such as more GT classes/labels)
            if fields_to_keep != None:
                for field_key in fields_to_keep:
                    func_name = fields_to_keep[field_key]
                    if func_name == 'max':
                        agg_dict[field_key] = lambda x: max(x)
                    elif func_name == 'min':
                        agg_dict[field_key] = lambda x: min(x)
                    else:
                        raise Exception('Aggregation function not implemented: {}'.format(func_name))
            if split_key in df and split_key not in agg_dict:
                agg_dict[split_key] = lambda x: max(x)
            if fold_key in df and fold_key not in agg_dict:
                agg_dict[fold_key] = lambda x: max(x)


            #Mark every group of close visits by a unique index. We sort the df by patients and visits, so consequtive
            # in time visits of a patient are also consecutive in the df. We calculate time delta between consecutive
            # visits, and when the time delta is GREATER than same_visit_date_diff, we mark that line with 1
            # (this'll be the start of a new index). We then cumsum these values to derive the actual indices
            df = df.sort_values(by=[patient_id_key, date_key])
            # For some reason, not all systems/pandas versions allow using .diff() on datetime objects, so I have to
            # translate datetime to integers:
            tmp = df[date_key].apply(lambda x: x.year*365+x.month*31+x.day*1 if(pd.notnull(x)) else x)
            tmp2 = tmp.diff().fillna(100).astype(int)
            tmp3 = tmp2.apply(lambda y: 1 if y > same_visit_date_diff else 0)
            df['visit_ind'] = tmp3.cumsum()
            # tmp = df.groupby([patient_id_key], as_index=False).apply(lambda x: x.groupby([date_key], as_index=False).agg(agg_dict).reset_index()).reset_index()
            # tmp2 = df.groupby([patient_id_key], as_index=False).apply(lambda x: x.groupby([date_key], as_index=False).agg(agg_dict))
            # tmp3 = df.groupby([patient_id_key]).apply(
            #     lambda x: x.groupby(x[date_key].diff().apply(lambda y: y.days).ge(2.0).cumsum(), as_index=False).agg(agg_dict))
            # tmp4 = df.groupby([patient_id_key]).apply(
            #     lambda x: x.groupby(x['visit_diff'].cumsum(), as_index=False).agg(
            #         agg_dict).reset_index()).reset_index()
            # tmp4 = df.groupby([patient_id_key]).apply(
            #     lambda x: x.groupby(x[date_key].diff().ge(2), as_index=False).agg(
            #         agg_dict).reset_index()).reset_inde
            # aggregate by patient id and close visits:
            df = df.groupby([patient_id_key, 'visit_ind']).agg(agg_dict).reset_index()

            # We now have a table where every visit has a line, containing a list of diagnoses ending with end_of_sentence_sep
            df = df.sort_values(by=[patient_id_key, date_key])

            df[age_key] = df.apply(lambda x: [x[age_key]] * len(x[diagnosis_col_new]), axis=1)
            df[age_month_key] = df.apply(lambda x: [x[age_month_key]] * len(x[diagnosis_col_new]), axis=1)

            if self.main_df is None:
                self.main_df = df
            else:
                self.main_df = self.main_df.append(df)
                self.main_df = self.main_df.sort_values(by=[patient_id_key, date_key])

    def apply_split(self, splitter):
        """
        splits the data in the container and returns a dictionary containing an input container with data from a split for every split name,
        :param splitter:
        :return:
        """
        split_dfs = splitter.apply_split(self.main_df)
        split_input_containers = {
            k: InputDataContainer(output_naming_conventions=self.output_naming_conventions, init_df=split_dfs[k]) for k
            in split_dfs}
        for k in split_input_containers:
            split_input_containers[k].corpus = self.corpus
            split_input_containers[k].max_age = self.max_age
        return split_input_containers

    def apply_time_filter(self, filt):
        """

        :param filt:
        """
        self.main_df = filt.apply_filter(self.main_df)

    def vectorize_inputs(self, output_naming_conventions, training_type, mode=None, min_visits=0):
        """
        Turns each visit in the container into a trajectory of visits (all the ones preceding it and the visit itself)
        We assume that the visits are sorted by patient ID and visit date (as they are during ingestion)
        :param mode: Whether to keep all visit trajectories ('all'), just the last one ('last'), all but the last one ('nolast'), or random (not implemented, TODO:)
        :param min_visits: trajectories with fewer visits are ignored
        :return:
        """
        patient_id_key = output_naming_conventions['patient_id_key']  # "ENROLID"
        age_key = output_naming_conventions['age_key']  # "AGE"
        age_month_key = output_naming_conventions['age_month_key']  # 'AGE_MON'
        dxver_key = output_naming_conventions['dxver_key']  # "DXVER" Diagnosis code version


        min_visits_per_patient = min_visits

        diagnosis_col_new = output_naming_conventions['diagnosis_vec_key']
        diagnosis_col_single_visit_new = "DXVIS"  # TODO: add these to naming conventions
        diagnosis_next_visit = 'DXNEXTVIS'


        if mode == 'all' or mode == 'nolast' or mode == 'last' or mode == 'random':
            self.main_df[diagnosis_col_single_visit_new] = self.main_df[diagnosis_col_new].apply(lambda x: x[:-1])
            self.main_df['visit_len'] = self.main_df[diagnosis_col_new].apply(lambda x: len(x))
            self.main_df['visit_len_cum'] = self.main_df.groupby(patient_id_key)['visit_len'].cumsum()

            #remove all patients with less than min_visits_per_patient
            self.main_df = self.main_df.groupby(patient_id_key).filter(lambda x: len(x) >= min_visits_per_patient)

            # generate for each visit the cumulative vector of all visits prior to it (separated by 'SEP' from before)
            # this would've been much easier with .groupby[].cumsum(), if not for a bug in cumsum: https://github.com/pandas-dev/pandas/issues/44009
            self.main_df[diagnosis_col_new] = self.main_df.groupby(patient_id_key)[diagnosis_col_new].transform('sum')
            self.main_df[diagnosis_col_new] = self.main_df.apply(lambda x: x[diagnosis_col_new][:x['visit_len_cum']],
                                                                 axis=1)
            self.main_df[age_key] = self.main_df.groupby(patient_id_key)[age_key].transform('sum')
            self.main_df[age_key] = self.main_df.apply(lambda x: x[age_key][:x['visit_len_cum']], axis=1)
            self.main_df[age_month_key] = self.main_df.groupby(patient_id_key)[age_month_key].transform('sum')
            self.main_df[age_month_key] = self.main_df.apply(lambda x: x[age_month_key][:x['visit_len_cum']], axis=1)

            # TODO: This part (next three lines) goes wrong if we use procedures as part of an input.
            #  For now I tried (without success) patching it by transforming all dx version values to strings
            #  (as they should have been), because some of them seem to be integers. I should check why this happens.
            # self.main_df[dxver_key] = self.main_df.apply(lambda x: str(x))
            # self.main_df[dxver_key] = self.main_df.groupby(patient_id_key)[dxver_key].transform('sum')
            # self.main_df[dxver_key] = self.main_df.apply(lambda x: x[dxver_key][:x['visit_len_cum']], axis=1)

            self.main_df[diagnosis_next_visit] = self.main_df[diagnosis_col_single_visit_new].shift(-1)
            self.main_df = self.main_df.drop(axis=0, columns=['visit_len_cum', 'visit_len'], errors='ignore')

            # mark next visit of the last visit for every patient as None
            grouped_by_PID = self.main_df.groupby(patient_id_key)
            self.main_df[diagnosis_next_visit][grouped_by_PID.cumcount(ascending=False) == 0] = None

            self.main_df['len_nextvis'] = self.main_df[diagnosis_next_visit].apply(
                lambda x: len(x) if (x is not None) else None)

        if training_type == 'nolast':
            # This should drop all visits that have no next visit
            self.main_df = self.main_df[self.main_df[diagnosis_next_visit] != None]
        elif training_type == 'last':
            # This should only keep the trajectory of a visit that has no next visit
            self.main_df = self.main_df[self.main_df[diagnosis_next_visit] == None]
        elif training_type == 'random':
            raise Exception('not implemented')

    def apply_gt(self, gt_container):
        self.main_df, added_gt_events = gt_container.add_gt_values_to_external_df(ext_df=self.main_df)

    def output_data(self, fpath):
        self.main_df = self.main_df.reset_index()
        output = {"max_age": self.max_age, "data_df": self.main_df}
        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))
        with open(fpath, 'bw') as f:
            pickle.dump(output, f)
        self.main_df.to_csv(fpath.replace('.pkl', '.csv'), index=False)


class DBExtractor():
    db_connection_params = None  # parameters for db2 connection
    gt_containers = None  # instance of the GTContainer class, to ingest external GT, contains all of the raw data, using output_naming_conventions
    data_container_inst = None  # instance of the InputDataContainer class, to ingest input data
    output_naming_conventions = None  # dictionary of names used in the internal db
    task = None
    columns_to_keep = None
    split_data_containers = None  # dictionary of {set_name: DataContainer}, where set_name is one of ('train', 'validation', 'test')

    def __init__(self, output_naming_conventions: dict, columns_to_keep: dict, task: str, gt_target_keys: list, gt_modes: dict = None):
        """

        :param db_connection_params:


        :param output_naming_conventions: naming conventions of the internal data representation, that will be utils to
        all functions that access the date (statistics, analysis, ML)
        :param: gt_modes: if str_external_gt is not None, this sets the mode of external GT (which time window to consider for event GT)

        """
        self.gt_containers = {gt_target_key: GTContainer(output_naming_conventions=output_naming_conventions,
                                                    modes=gt_modes[gt_target_key], target_key=gt_target_key) for gt_target_key in gt_target_keys}
        self.data_container_inst = InputDataContainer(output_naming_conventions=output_naming_conventions)
        self.output_naming_conventions = output_naming_conventions
        self.columns_to_keep = columns_to_keep
        self.task = task
        a = 1

    def translate_df(self, df, input_naming_conventions):
        """
        Translates input df, with columns named as described in input_naming_conventions to internal naming conventions
        as described in self.output_naming_conventions
        :type df: pandas.core.frame.DataFrame
        :param df:
        :type input_naming_conventions: dict
        :param input_naming_conventions:
        :return: translated dataframe
        """
        validate_naming_conventions(input_naming_conventions)
        data_column_renames = {
            input_naming_conventions['patient_id_key']: self.output_naming_conventions['patient_id_key'],
            input_naming_conventions['index_date_key']: self.output_naming_conventions['index_date_key'],
            input_naming_conventions['svc_date_key']: self.output_naming_conventions['svc_date_key'],
            input_naming_conventions['adm_date_key']: self.output_naming_conventions['adm_date_key'],
            input_naming_conventions['age_key']: self.output_naming_conventions['age_key'],
            # input_naming_conventions['age_month_key']   :   self.output_naming_conventions['age_month_key'],
            input_naming_conventions['date_birth_key']: self.output_naming_conventions['date_birth_key'],
            input_naming_conventions['gender_key']: self.output_naming_conventions['gender_key'],
            input_naming_conventions['dxver_key']: self.output_naming_conventions['dxver_key'],
            input_naming_conventions['outcome_key']: self.output_naming_conventions['outcome_key'],
            input_naming_conventions['split_key']: self.output_naming_conventions['split_key'],
            input_naming_conventions['fold_key']: self.output_naming_conventions['fold_key'],
            # input_naming_conventions['outcome_key']: self.output_naming_conventions['outcome_key']
        }
        out_df = df.rename(columns=data_column_renames, inplace=False)
        return out_df

    def ingest_sql(self, db_connection_params, input_naming_conventions, event_naming_conventions, output_naming_conventions, 
                   str_external_gt=None, gt_target_key=None, str_data=None, same_visit_date_diff=1, data_source='sql_db'):
        """

        :type db_connection_params: dict
        :type input_naming_conventions: dict
        :type str_external_gt: str
        :type str_data: str
        :param str_data: sql code to extract the relevant table
        :param db_connection_params: parameters for connection to a relevant db
        :param event_naming_conventions: naming conventions for event dataset
        :param input_naming_conventions: A dictionary that describes the meaning of fields in the input db.
        Must contain the following fields:
                'patient_id_key' (e.g. 'ENROLID'): (string) Unique patient ID
                'index_date_key' (e.g. 'INDDATE'): (datetime) Index date of the disease - first diagnosis or first treatment (earlier of the two)
                'svc_date_key' (e.g. 'SVCDATE'): (datetime) Date of service (there may be several services per admission)
                'adm_date_key' (e.g. 'ADMDATE'): (datetime) Date of admission to a hospital (in case of a single day visit, e.g. for outpatients, this is the same as service date)
                'age_key' (e.g. 'AGE'): (int) Age in years
                'age_month_key' (e.g. 'AGE_MON'): (int)(optional) age in months. If doesn't exist, will be recreated using service/admission date and year of birth.
                'date_birth_key' (e.g. 'DOBYR'): Year of birth
                'gender_key' (e.g. 'SEX'): (string) Patient sex at birth(?)
                'dxver_key' (e.g. 'DXVER'): (string) Diagnosis code version (e.g. ICD9 or ICD10)
                'outcome_key' (e.g. 'PD'): (int 0 or 1) Indicator of whether or not the patient is part of the sick cohort
                'input_columns' (e.g. ['DX1', 'DX2', 'PROC1']) Names of columns that contain input information - diagnoses, procedures, etc.
                            Diagnosis codes are left as is (strings), procedure strings (i.e. columns whose name contains 'proc') will have 'pr_' appended to their values by the function
        :param str_external_gt: optional external event gt data (patient/date/event)
        :param same_visit_date_diff: if difference between consecutive visits is less than or equal to so many days, they're considered the same visit
        :param data_source: from where to read the data: 'sql_db' or 'csv'
        :return:
        """
        if data_source == 'csv':
            if str_data != None:
                headers = ['col1', 'col2', 'col3', 'col4']
                dtypes = {'col1': 'str', 'col2': 'str', 'col3': 'str', 'col4': 'float'}
                parse_dates = [input_naming_conventions['adm_date_key'], input_naming_conventions['svc_date_key'], input_naming_conventions['index_date_key']]
                data_df = pd.read_csv(str_data, sep=',',  parse_dates=parse_dates)
                data_df = self.translate_df(data_df, input_naming_conventions)
                # Vadim:
                self.data_container_inst.ingest_df(data_df, task=self.task, fields_to_keep=self.columns_to_keep,
                                                   input_columns=input_naming_conventions['input_columns'],
                                                   output_naming_conventions=output_naming_conventions,
                                                   same_visit_date_diff=same_visit_date_diff)
            if str_external_gt != None:
                gt_df = pd.read_csv(str_data)
                self.gt_containers[gt_target_key].add_event_df(df=gt_df,
                                                               event_names=None,
                                                               # use all event types present in the df
                                                               event_naming_conventions=event_naming_conventions,
                                                               same_visit_date_diff=same_visit_date_diff)
        else:
            raise Exception(f'Data source {data_source} not implemented')


    def apply_split(self, splitter):
        """
        :param splitter: SplitFilter instance, containing instructions to split the df in container
        :return: None
        """
        self.split_data_containers = self.data_container_inst.apply_split(splitter)

    def apply_time_filter(self, tr_set, time_filter):
        """
        Filters a tr_set (split: 'train', 'validation', 'test' or all data: 'all') using a time filter
        :param tr_set: which set to filter - 'train', 'validation', 'test' (one of the keys in self.split_data_containers)  or all data: 'all'
        :param time_filter: TimeFilter instance
        :return: None
        """
        if tr_set != 'all':
            assert tr_set in self.split_data_containers
            self.split_data_containers[tr_set].apply_time_filter(time_filter)
        else:
            self.data_container_inst.apply_time_filter(time_filter)

    def vectorize_inputs(self, output_naming_conventions, training_type, tr_set: str, keep_trajectories: str=None, min_visits: int = 0):
        """
        Turns each visit in a tr_set into a trajectory of visits (all the ones preceding it and the visit itself)
        :param tr_set: which set to filter - 'train', 'validation', 'test' (one of the keys in self.split_data_containers)  or all data: 'all'
        :param keep_trajectories: Whether to keep all visit trajectories ('all'), just the last one ('last'), or random (not implemented, TODO:)
        :param min_visits: minimum number of visits in a trajectory (trajectories with fewer visits are ignored)
        :return:
        """
        mode = keep_trajectories
        if tr_set != 'all':
            assert tr_set in self.split_data_containers
            self.split_data_containers[tr_set].vectorize_inputs(output_naming_conventions, training_type, mode, min_visits)
        else:
            self.data_container_inst.vectorize_inputs(output_naming_conventions, training_type, mode, min_visits)

    def apply_gt(self, tr_set: str):
        """
        Turns each visit in a tr_set into a trajectory of visits (all the ones preceding it and the visit itself)
        :param tr_set: which set to filter - 'train', 'validation', 'test' (one of the keys in self.split_data_containers)  or all data: 'all'
        :return:
        """
        for gt_target_key in self.gt_containers:
            if tr_set != 'all':
                assert tr_set in self.split_data_containers
                self.split_data_containers[tr_set].apply_gt(self.gt_containers[gt_target_key])
            else:
                self.data_container_inst.apply_gt(self.gt_containers[gt_target_key])

    def output_to_files(self, file_config: dict, tr_val_tst_names:list, tr_val_test_dirs: list, format: str = 'BEHRT'):
        """
        Saves the inner dataset to files
        :param tr_val_test_names: list of train, val and test keys in self.split_data_containers
        :param tr_val_tst_dirs: list or train, val and test path keys in file_config
        :param format: 'BEHRT', 'PYHEALTH' (not implemented)
        :param file_config: dictionary containing output paths:
            'vocab': '/data/usr/vadim/EHR/PD_ALL_SERV_200000_0_to_ind_90_to_event/EVENT/PD_small.vocab', - input vocabulary path (string to integer)
            'gt_vocab': '/data/usr/vadim/EHR/PD_ALL_SERV_200000_0_to_ind_90_to_event/EVENT/PD_gt_small.vocab', gt vocabulary path
            'train': '/data/usr/vadim/EHR/PD_ALL_SERV_200000_0_to_ind_90_to_event/EVENT/train/PD', train set input path
            'test': '/data/usr/vadim/EHR/PD_ALL_SERV_200000_0_to_ind_90_to_event/EVENT/test/PD', test set
            'val': '/data/usr/vadim/EHR/PD_ALL_SERV_200000_0_to_ind_90_to_event/EVENT/val/PD' val set
        :return:
        """
        assert format == 'BEHRT' # other options not implemented yet

        vocab_out_path = file_config['vocab']  # .path.join(data_main_path, training_type, task + "_small.vocab.pkl")
        corpus_out_path = vocab_out_path + '.corpus'

        for gt_event_key in self.gt_containers:
            vocab_gt_out_path = get_vocab_path(root_vocab_path=file_config['gt_vocab'], event_type_identifier=gt_event_key)
            corpus_gt_out_path = vocab_gt_out_path + '.corpus'
            self.gt_containers[gt_event_key].output_vocab(corpus_path=corpus_gt_out_path, vocab_path=vocab_gt_out_path)

        self.data_container_inst.output_vocab(corpus_path=corpus_out_path, vocab_path=vocab_out_path)


        for tr_dir, tr_name in zip(tr_val_test_dirs, tr_val_tst_names):
            out_df_path = file_config[tr_dir] + '.pkl'
            self.split_data_containers[tr_name].output_data(fpath=out_df_path)


def transform_data(config_file: str):
    argv = ['transform_data', config_file]
    if len(argv) > 1:
        config_file_path = argv[1]  
    else:
        config_file_path = None

    if len(argv) > 1:
        config_file_path = argv[1]
    else:
        config_file_path = None
    host_name = socket.gethostname()
    global_params, file_config, model_config, optim_config, data_config, output_naming_conventions = get_config(
        config_file_path)

    days_to_inddate = data_config['days_to_inddate']  # 180 #None #180
    limit_visits = data_config[
        'limit_visits']  # 50000000 #10000 #None #don't take less than 10k, otherwise you'll get only a single visit per patient, which is not enough in case of next visit prediction
    x_sub = "x_data"
    y_sub = "y_data"
    task = data_config['task']
    # task_short = 'PD'
    # training_type = 'LABEL_NO_LAST' #removing last visit for next visit prediction
    training_type = 'LABEL'  # keeping all visits for outcome predictions
    use_external_gt = True
    sql_select_gt = None

    if data_config['data_source'] == 'csv':
        if limit_visits is None:
            sql_select = [data_config['data_source_str']]
            sql_select_gt = data_config['gt_source_str']
        else:
            sql_select = [data_config['sample_data_source_str']]
            sql_select_gt = data_config['gt_source_str']        
    else:
        raise Exception(f'Data source {data_config["data_source"]} not implemented')
    
    do_write_files = True  # False #True
    
    if do_write_files:        
        db_cnxn_params = None

        naming_conventions_event = { output_naming_conventions['event_key']: { #TODO: move this to config
                                                                                'patient_id_key': 'ENROLID',
                                                                                'event_type_key': 'EVENT_TYPE',
                                                                                'event_date_key': 'EVENT_DATE',
                                                                                'event_code_key': 'CODE',
                                                                                'event_code_type_key': 'CODE_TYPE'
                                                                            },
                                    output_naming_conventions['treatment_event_key'] : { #TODO: move this to config
                                                                                'patient_id_key': 'ENROLID',
                                                                                'event_type_key': 'STR_INGRD',
                                                                                'event_date_key': 'SVCDATE',
                                                                                # 'event_code_key': 'RXCUI_INGRD',
                                                                                # 'event_code_type_key': None
                                                                            },
                                    }
        gt_modes = {output_naming_conventions['event_key']: {'all': {'future_timeframe': data_config['event_prediction_window_days']}}, #'within_90_days',
                    output_naming_conventions['treatment_event_key'] : {'all': {'future_timeframe': data_config['treatment_event_prediction_window_days']}},
        }
        if sql_select_gt != None:
            try:
                gt_modes = {k: gt_modes[k] for k in sql_select_gt}
            except:
                raise Exception("not all keys in sql_sel_gt have a defined gt_mode")
            try:
                naming_conventions_event = {k: naming_conventions_event[k] for k in sql_select_gt}
            except:
                raise Exception("not all keys in sql_sel_gt have a defined naming convention")
        # for m in external_gt_types:
        #     gt_modes[m] = {'future_timeframe': event_prediction_window_days}

        input_column_names = data_config['input_column_names'] #["DX1", "DX2", "DX3", "DX4"]
        if data_config['use_procedures']:
            input_column_names.append("PROC1")

        in_db_naming_conventions = { #TODO: move this to config
            'patient_id_key': 'ENROLID',
            'index_date_key': 'INDDATE',
            'svc_date_key': 'SVCDATE',
            'adm_date_key': 'ADMDATE',
            'age_key': 'AGE',
            'age_month_key': 'AGE_MON',
            'date_birth_key': 'DOBYR',
            'dxver_key': 'DXVER',
            'outcome_key': 'PD',
            'event_type_key': 'EVENT_TYPE',
            # (string)(optional, if additional GT table is used) general groups of events (as opposed to specific diagnostic codes) - these will most likely be the ones we'll try to predict
            'event_date_key': 'EVENT_DATE',  # (string)(optional, if additional GT table is used) date of a given event
            'event_code_key': 'CODE',  # (string) specific event diagnostic code
            'event_code_type_key': 'CODE_TYPE',  # (string) category of event
            'gender_key': 'SEX',
            'split_key': 'SPLIT',
            'fold_key': 'FOLD',
            'input_columns': input_column_names,
        }
        train_name = 'train'
        val_name = 'validation'
        test_name = 'test'

        columns_to_keep = {output_naming_conventions['outcome_key']: 'max',
                            # output_naming_conventions['fold_key']: 'max',
                            output_naming_conventions['split_key']: 'max'}
        
        extractor_instance = DBExtractor(output_naming_conventions=output_naming_conventions,
                                          columns_to_keep=columns_to_keep, gt_target_keys=naming_conventions_event.keys(), task=task, gt_modes=gt_modes)
        # ingest data:
        for sel in sql_select:
            # template_params = {}
            # templating_engine = TemplatingEngine(template_params)
            # tdc = TemplatedDatabaseConnector(cnxn=cnxn, templating_engine=templating_engine)
            extractor_instance.ingest_sql(db_connection_params=db_cnxn_params,
                                          input_naming_conventions=in_db_naming_conventions, event_naming_conventions=None, output_naming_conventions=output_naming_conventions, str_data=sel,
                                          str_external_gt=None, gt_target_key=None, same_visit_date_diff=data_config['visit_days_resolution'], data_source=data_config['data_source'])

        # split data to train val test sets:
        splitter = SplitFilter(split_method='in_data',
                               tr_val_tst_names=[train_name, val_name, test_name],
                               # TODO: Add the names to a config. It's important, because the names are also external - these are the values that should be in every input DB 'SPLIT' column
                               tr_val_tst_ratios=None,
                               tr_val_tst_column=output_naming_conventions['split_key'],
                               split_dict=None,
                               split_level_key=output_naming_conventions['patient_id_key'])
        extractor_instance.apply_split(splitter)
        # optional: filter individual visits here
        if 'days_to_inddate_tr' in data_config:
            time_stop = data_config['days_to_inddate_tr']
        else:
            time_stop = data_config['days_to_inddate']
        if 'days_to_inddate_start_tr' in data_config:
            time_start = data_config['days_to_inddate_start_tr']
        else:
            time_start = data_config['days_to_inddate_start']

        # Filter visits by time (this will help to reduce the number of trajectories later, and reduce trajectory computation time:
        train_time_filt = TimeFilter(start_days_before=None, stop_days_before=time_stop,
                                     date_column_name=output_naming_conventions['date_key'],
                                     ref_date_column_name=output_naming_conventions['index_date_key'])
        extractor_instance.apply_time_filter(train_name, train_time_filt)

        time_stop = data_config['days_to_inddate_tr']
        time_start = data_config['days_to_inddate_start_tr']
        test_time_filt = TimeFilter(start_days_before=time_start, stop_days_before=time_stop,
                                    date_column_name=output_naming_conventions['date_key'],
                                    ref_date_column_name=output_naming_conventions['index_date_key'])
        extractor_instance.apply_time_filter(test_name, test_time_filt)
        extractor_instance.apply_time_filter(val_name, test_time_filt)

        # create input vectors
        extractor_instance.vectorize_inputs(output_naming_conventions, training_type, tr_set=train_name, keep_trajectories='all', min_visits=data_config['min_visits_per_patient'])
        extractor_instance.vectorize_inputs(output_naming_conventions, training_type, tr_set=test_name, keep_trajectories='all', min_visits=data_config['min_visits_per_patient'])
        extractor_instance.vectorize_inputs(output_naming_conventions, training_type, tr_set=val_name, keep_trajectories='all', min_visits=data_config['min_visits_per_patient'])

        # Filter trajectories by time:
        train_time_filt = TimeFilter(start_days_before=time_start, stop_days_before=time_stop,
                                     date_column_name=output_naming_conventions['date_key'],
                                     ref_date_column_name=output_naming_conventions['index_date_key'])
        extractor_instance.apply_time_filter(train_name, train_time_filt)

        time_stop = data_config['days_to_inddate']
        time_start = data_config['days_to_inddate_start']
        test_time_filt = TimeFilter(start_days_before=time_start, stop_days_before=time_stop,
                                    date_column_name=output_naming_conventions['date_key'],
                                    ref_date_column_name=output_naming_conventions['index_date_key'])
        extractor_instance.apply_time_filter(test_name, test_time_filt)
        extractor_instance.apply_time_filter(val_name, test_time_filt)

        # ingest and apply event GT to train val and test sets:
        if sql_select_gt != None:
            for sel_gt_key in sql_select_gt:
                for sel_gt in sql_select_gt[sel_gt_key]:
                    extractor_instance.ingest_sql(db_connection_params=db_cnxn_params,
                                                  input_naming_conventions=in_db_naming_conventions, event_naming_conventions=naming_conventions_event[sel_gt_key], output_naming_conventions=output_naming_conventions,
                                                  str_data=None, str_external_gt=sel_gt, gt_target_key=sel_gt_key,data_source=data_config['data_source'])
            extractor_instance.apply_gt(tr_set=train_name)
            extractor_instance.apply_gt(tr_set=val_name)
            extractor_instance.apply_gt(tr_set=test_name)

        # write the relevant datasets and vocabularies to files:
        extractor_instance.output_to_files(file_config=file_config, tr_val_test_dirs=['train', 'val', 'test'], tr_val_tst_names=[train_name, val_name, test_name], format='BEHRT')


if __name__ == "__main__":
    transform_data(os.path.dirname(os.path.abspath(__file__))+'/multi_config_CKD.yaml')