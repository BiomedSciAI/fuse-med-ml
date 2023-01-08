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


class StatsAnalyzer:
    data_df = None
    title = ""

    def __init__(self, input_df, naming_conventions, title=""):
        """
        Creates a new StatsAnalyzer instance
        :param naming_conventions: names of the relevant fields of the input df
        :param input_df: the dataframe that contains patient data to run statistics on. We'll store a
        copy of it, so as not to change the original.
        :param title: Title of the dataset (usually 'train'/'val'/'test', etc.)

        naming_config (for reference, defined in ehrtransformers.configs.config)
        'diagnosis_vec_key': 'DX',
        'age_key': 'AGE',
        'age_month_key': 'AGE_MON',
        'label_key': 'label',
        'date_key': "ADMDATE",
        'patient_id_key': "ENROLID",
        'outcome_key': "GT",
        'disease_key': 'PD',
        'gender_key': "SEX",
        'separator_str' : 'SEP',
        'sick_val' : '1',
        'healthy_val' : '0'
        """

        self.data_df = input_df.copy(deep=True)
        self.naming = naming_conventions
        self.title = title

        self.orig_line_num = input_df.shape[0]
        # compute the number of visits per row:
        self.data_df["n_visits"] = self.data_df[self.naming["diagnosis_vec_key"]].apply(
            lambda x: x.count(self.naming["separator_str"])
        )

        # leave only the last visit vector:
        idx = (
            self.data_df.groupby([self.naming["patient_id_key"]])["n_visits"].transform(max) == self.data_df["n_visits"]
        )
        self.data_df = self.data_df[idx]

        # add 'LAST_AGE' and 'FIRST_AGE' columns containing ages during last and first visits of a patient
        if self.naming["age_month_key"] in self.data_df:
            self.data_df["LAST_AGE"] = self.data_df[self.naming["age_month_key"]].apply(lambda x: float(x[-1]) / 12)
            self.data_df["FIRST_AGE"] = self.data_df[self.naming["age_month_key"]].apply(lambda x: float(x[0]) / 12)
        else:
            self.data_df["LAST_AGE"] = self.data_df[self.naming["age_key"]].apply(lambda x: float(x[-1]) / 12)
            self.data_df["FIRST_AGE"] = self.data_df[self.naming["age_key"]].apply(lambda x: float(x[0]) / 12)
        a = 1

    def get_patient_num(self, sick_healthy="all"):
        """
        :param sick_healthy:  'sick', 'healthy', 'all' - positive, negative or all patients. TODO: add a dictionary parameter that allows specific column/value analysis
        :return: number of patients in the input dataframe.
        """
        if sick_healthy == "all":
            patient_list = set(self.data_df[self.naming["patient_id_key"]])
        elif sick_healthy == "sick":
            patient_list = set(
                self.data_df[self.data_df[self.naming["disease_key"]] == self.naming["sick_val"]][
                    self.naming["patient_id_key"]
                ]
            )
        elif sick_healthy == "healthy":
            patient_list = set(
                self.data_df[self.data_df[self.naming["disease_key"]] == self.naming["healthy_val"]][
                    self.naming["patient_id_key"]
                ]
            )
        else:
            raise Exception("Unexpected sick/healthy value of {}".format(sick_healthy))

        return len(patient_list)

    def get_visit_num(self, sick_healthy="all"):
        """
        Computes information on number of visits. Works on the input dataframe, where only a single line was left per patient - the one with most visits.
        :param sick_healthy: 'sick', 'healthy', 'all' - positive, negative or all patients. TODO: add a dictionary parameter that allows specific column/value analysis
        :return: total visits (i.e. sum of visit numbers per patient), mean visits (per patient), per-patient visit std
        """
        if sick_healthy == "all":
            tmp_data = self.data_df
        else:
            if sick_healthy == "sick":
                disease_gt = self.naming["sick_val"]
            elif sick_healthy == "healthy":
                disease_gt = self.naming["healthy_val"]
            else:
                raise Exception("Unexpected sick/healthy value of {}".format(sick_healthy))
            tmp_data = self.data_df[self.data_df[self.naming["disease_key"]] == disease_gt]
        total_visits = tmp_data["n_visits"].sum()
        mean_visits = tmp_data["n_visits"].mean()
        std_visits = tmp_data["n_visits"].std()
        return total_visits, mean_visits, std_visits

    def get_patient_ages(self, sick_healthy="all"):
        """
        Computes information on patient ages, at the time of the last visit. Works on the input dataframe, where only a single line was left per patient - the one with most visits.
        :param sick_healthy: 'sick', 'healthy', 'all' - positive, negative or all patients. TODO: add a dictionary parameter that allows specific column/value analysis
        :return: total visits (i.e. sum of visit numbers per patient), mean visits (per patient), per-patient visit std
        """
        if sick_healthy == "all":
            tmp_data = self.data_df
        else:
            if sick_healthy == "sick":
                disease_gt = self.naming["sick_val"]
            elif sick_healthy == "healthy":
                disease_gt = self.naming["healthy_val"]
            else:
                raise Exception("Unexpected sick/healthy value of {}".format(sick_healthy))
            tmp_data = self.data_df[self.data_df[self.naming["disease_key"]] == disease_gt]

        mean_last_age = tmp_data["LAST_AGE"].mean()
        mean_first_age = tmp_data["FIRST_AGE"].mean()
        std_last_age = tmp_data["LAST_AGE"].std()
        std_first_age = tmp_data["FIRST_AGE"].std()
        # total_visits = tmp_data['n_visits'].sum()
        # mean_visits = tmp_data['n_visits'].mean()
        # std_visits = tmp_data['n_visits'].std()
        return mean_last_age, std_last_age, mean_first_age, std_first_age

    def get_pre_inddate_delta(self):
        """
        Computes differences between last visit and inddate (min, max, mean, stdev)
        :return:
        """
        # TODO: implement
        a = 1

    def get_overlapping_patients(self, input_df):
        """

        :param input_df: input dataframe to compare with the df the analyzer was initialized with
        :return: a list of overlapping patients between the init dataframe and the compared dataframe
        """
        p1 = set(list(self.data_df[self.naming["patient_id_key"]]))
        p2 = set(list(input_df[self.naming["patient_id_key"]]))
        return p1.intersection(p2)

    def print_stats(self):
        """
        Prints statistics of the input dataframe of patient EHR information to stdout.
        :return:
        """
        print("Statistics for {} dataset:".format(self.title))
        print("Total Entries: {}".format(self.orig_line_num))
        print("Patient info:")
        print("{:>15}~{:>8}~{:>8}".format("Patient number", "Sick", "Healthy"))
        print(
            "{:15}~{:8}~{:8}".format(
                self.get_patient_num("all"), self.get_patient_num("sick"), self.get_patient_num("healthy")
            )
        )
        print("Visit number info:")
        print("{:<10}~{:>10}~{:>20}~{:>10}".format("group", "total", "mean per patient", "stdev"))
        for grp in ["all", "sick", "healthy"]:
            visits, vis_mean, vis_std = self.get_visit_num(grp)
            print("{:<10}~{:10}~{:20.2f}~{:10.2f}".format(grp, visits, vis_mean, vis_std))

        print("Patient age info:")
        print("{:<10}~{:>10}~{:>10}~{:>10}~{:>10}".format("group", "mean last", "std last", "mean first", "std first"))
        for grp in ["all", "sick", "healthy"]:
            mean_last_age, std_last_age, mean_first_age, std_first_age = self.get_patient_ages(grp)
            print(
                "{:<10}~{:10.2f}~{:10.2f}~{:10.2f}~{:10.2f}".format(
                    grp, mean_last_age, std_last_age, mean_first_age, std_first_age
                )
            )
