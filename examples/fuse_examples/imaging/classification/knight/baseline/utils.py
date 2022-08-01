import os
import json
import pandas as pd
import numpy as np

CLINICAL_NAMES = [
    "SubjectId",
    "age",
    "bmi",
    "gender",
    "gender_num",
    "comorbidities",
    "smoking_history",
    "radiographic_size",
    "preop_egfr",
    "pathology_t_stage",
    "pathology_n_stage",
    "pathology_m_stage",
    "grade",
    "aua_risk_group",
    "task_1_label",
    "task_2_label",
]


def create_knight_clinical(original_file, processed_file=None):
    with open(original_file) as f:
        clinical_data = json.load(f)
    t_stage_count = np.zeros((5))
    aua_risk_count = np.zeros((5))
    df = pd.DataFrame(columns=CLINICAL_NAMES)
    for index, patient in enumerate(clinical_data):
        df.loc[index, "SubjectId"] = patient["case_id"]
        df.loc[index, "age"] = patient["age_at_nephrectomy"]
        df.loc[index, "bmi"] = patient["body_mass_index"]

        df.loc[index, "gender"] = patient["gender"]
        if patient["gender"] == "male":  # 0:'male'  1:'female','transgender_male_to_female'
            df.loc[index, "gender_num"] = 0
        else:
            df.loc[index, "gender_num"] = 1

        df.loc[index, "comorbidities"] = 0  # 0:no_comorbidities 1:comorbidities_exist
        for key, value in patient["comorbidities"].items():
            if value:
                df.loc[index, "comorbidities"] = 1

        df.loc[index, "smoking_history"] = patient["smoking_history"]
        if patient["smoking_history"] == "never_smoked":  # 0:'never_smoked' 1:'previous_smoker'  2:'current_smoker'
            df.loc[index, "smoking_history"] = 0
        elif patient["smoking_history"] == "previous_smoker":
            df.loc[index, "smoking_history"] = 1
        elif patient["smoking_history"] == "current_smoker":
            df.loc[index, "smoking_history"] = 2

        df.loc[index, "radiographic_size"] = patient["radiographic_size"]
        if patient["last_preop_egfr"]["value"] == ">=90":
            df.loc[index, "preop_egfr"] = 90
        else:
            df.loc[index, "preop_egfr"] = patient["last_preop_egfr"]["value"]

        df.loc[index, "pathology_t_stage"] = patient["pathology_t_stage"]
        df.loc[index, "pathology_n_stage"] = patient["pathology_n_stage"]
        df.loc[index, "pathology_m_stage"] = patient["pathology_m_stage"]
        df.loc[index, "aua_risk_group"] = patient["aua_risk_group"]

        # Task 1 labels:
        if patient["aua_risk_group"] in ["high_risk", "very_high_risk"]:  # 1:'3','4'  0:'0','1a','1b','2a','2b'
            df.loc[index, "task_1_label"] = 1  # CanAT
        else:
            df.loc[index, "task_1_label"] = 0  # NoAT

        # Task 2 labels:
        if patient["aua_risk_group"] == "benign":
            df.loc[index, "task_2_label"] = 0
        elif patient["aua_risk_group"] == "low_risk":
            df.loc[index, "task_2_label"] = 1
        elif patient["aua_risk_group"] == "intermediate_risk":
            df.loc[index, "task_2_label"] = 2
        elif patient["aua_risk_group"] == "high_risk":
            df.loc[index, "task_2_label"] = 3
        elif patient["aua_risk_group"] == "very_high_risk":
            df.loc[index, "task_2_label"] = 4
        else:
            ValueError("Wrong risk class")

        # former classification - deprecated
        # if patient['pathology_t_stage'] in ['3', '4']:    # 1:'3','4'  0:'0','1a','1b','2a','2b'
        #    df.loc[index, 'pathology_t_stage_classify'] = 1
        # else:
        #    df.loc[index, 'pathology_t_stage_classify'] = 0
        t_stage = int(patient["pathology_t_stage"][0])
        t_stage_count[t_stage] += 1
        aua_risk = int(df.loc[index, "task_2_label"])
        aua_risk_count[aua_risk] += 1
        df.loc[index, "grade"] = patient["tumor_isup_grade"]

    if processed_file is not None:
        # save csv file
        df.to_csv(processed_file, index=False)
        df = df.drop(["gender", "pathology_t_stage", "pathology_n_stage", "pathology_m_stage"], axis=1)
        df.to_csv(os.path.splitext(processed_file)[0] + "_numeric.csv", index=False)
    print(f"Pathology t-stage count summary: {t_stage_count}")
    print(f"AUA risk count summary: {aua_risk_count}")
    return df
