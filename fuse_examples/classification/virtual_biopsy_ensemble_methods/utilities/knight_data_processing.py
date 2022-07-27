import os
import json
import pandas as pd
import numpy as np
import pickle as pkl


# Change here according to your local path where the clinical data is 
data_path = 'KNIGHT/knight/data/knight.json'


def read_json_file_to_df(filename):
    '''Reads json file and transforms into a pandas dataframe indexed by case_id.
    Args:
    filename (str): path to the json file.
    Returns (pd.DataFrame): data contained in json file in format of dataframe.'''

    f = open(filename)
    data = json.load(f)
    df = pd.DataFrame.from_dict(data)
    df = df.set_index('case_id')

    return df

def process_KNIGHT_clinical_data(data_path = data_path, 
                                 processed_file_name=None):
    '''Data processing of raw KNIGHT clinical data. 

    Args: 
    data_path (str): path to the json file containing KNIGHT data.
    processed_file_name (str): name of processed data in csv to be saved.
    
    Returns (pd.DataFrame): processed clinical data of the entire dataset'''
    
    df = read_json_file_to_df(data_path)
        
    # Gender
    gender_codes = {'male': 0, 'female': 1, 'transgender_male_to_female': 0, 'MALE': 0}
    df['gender'] = df['gender'].map(gender_codes)
    
    # Comorbidities
    df = df.merge(df['comorbidities'].apply(pd.Series).astype(int), right_index = True, left_index = True)
    df = df.drop (columns = 'comorbidities')
    
    # Smoking level: atribute value for each category (0, 1 or 2)
    smoking_codes = {'never_smoked': 0, 'previous_smoker': 1, 'current_smoker': 2}
    df['smoking_level'] = df['smoking_history'].map(smoking_codes)
    
    # Smoking history: if the patient ever smoked before (0 or 1)
    ever_smoked_codes = {'never_smoked': 0, 'previous_smoker': 1, 'current_smoker': 1}
    df['ever_smoked'] = df['smoking_history'].map(ever_smoked_codes)
    df = df.drop(columns = 'smoking_history')
    
    # Age when quit smoking
    df['age_when_quit_smoking'] = df['age_when_quit_smoking'].replace({'not_applicable': 100, '0': np.nan})
    df['age_when_quit_smoking'] = pd.to_numeric(df['age_when_quit_smoking'])
    
    # Chewing tobacco use
    chewing_tob_codes = {'never_or_not_in_last_3mo': 0, 'quit_in_last_3mo': 1, 'currently_chews': 2, ' ': np.nan}
    df['chewing_tobacco_use'] = df['chewing_tobacco_use'].map(chewing_tob_codes)
    
    # Alcohol use
    alcohol_codes = {'never_or_not_in_last_3mo': 0, 'two_or_less_daily': 1, 'more_than_two_daily': 2, 'quit_in_last_3mo': 3, ' ': np.nan}
    df['alcohol_use'] = df['alcohol_use'].map(alcohol_codes)
    
    # Last preoperative EGFR
    last_preop_egfr = df['last_preop_egfr'].apply(pd.Series, dtype='object') # transform into 2 columns
    last_preop_egfr.columns = ['last_preop_egfr_value', 'days_before_nephrectomy_egfr_was_measured']
    
    last_preop_egfr['last_preop_egfr_value'] = last_preop_egfr['last_preop_egfr_value'].replace({'>=90': 90, '>90':90})
    last_preop_egfr['last_preop_egfr_value'] = pd.to_numeric(last_preop_egfr['last_preop_egfr_value'])
    
    last_preop_egfr['days_before_nephrectomy_egfr_was_measured'] = last_preop_egfr['days_before_nephrectomy_egfr_was_measured'].replace({'>90': 90})
    
    df = df.merge(last_preop_egfr, right_index = True, left_index = True)
    df = df.drop (columns = 'last_preop_egfr') 
    
    # Create BMI category feature
    df.loc[df['body_mass_index'] <= 18.5,'BMI category'] = 0
    df.loc[(df['body_mass_index'] <= 24.9) & (df['body_mass_index'] > 18.8), 'BMI category'] = 1
    df.loc[(df['body_mass_index'] <= 29.9) & (df['body_mass_index'] > 24.9), 'BMI category'] = 2
    df.loc[df['body_mass_index'] > 24.9, 'BMI category'] = 3
       
    # Create labels of Adjuvant Therapy: 1 (candidate for Adjuvant Therapy: high risk and very high risk individuals), 0 otherwise
    df.loc[(df['aua_risk_group'].isin(['high_risk', 'very_high_risk'])), 'adj_therapy_label'] = 1
    df.loc[(df['aua_risk_group'].isin(['benign', 'low_risk', 'intermediate_risk'])), 'adj_therapy_label'] = 0
         
    # Create multiclass labels
    df.loc[(df['aua_risk_group'] == 'benign'), 'risk_label'] = 0
    df.loc[(df['aua_risk_group'] == 'low_risk'), 'risk_label'] = 1
    df.loc[(df['aua_risk_group'] == 'intermediate_risk'), 'risk_label'] = 2
    df.loc[(df['aua_risk_group'] == 'high_risk'), 'risk_label'] = 3
    df.loc[(df['aua_risk_group'] == 'very_high_risk'), 'risk_label'] = 4
    
    df['risk_label'] = df['risk_label'].astype(int)
    
    # Create dummy columns for multiclass labels
    dummy_labels = pd.get_dummies(df['risk_label'], prefix='risk_label')
    df = df.merge(dummy_labels, left_index = True, right_index = True )
    
    # Only use pre-operative clinical features (most of the  features in train set are not available for test set) + labels
    
    label_cols = [x for x in df if 'label' in x]
    df = df[['age_at_nephrectomy', 'gender', 'age_when_quit_smoking', 'chewing_tobacco_use', 'alcohol_use',
              'smoking_level', 'ever_smoked', 'radiographic_size',  'body_mass_index', 'last_preop_egfr_value',
              'days_before_nephrectomy_egfr_was_measured', 'BMI category', 'myocardial_infarction', 'congestive_heart_failure', 
              'peripheral_vascular_disease', 'cerebrovascular_disease', 'dementia', 'copd', 'connective_tissue_disease', 
              'peptic_ulcer_disease', 'uncomplicated_diabetes_mellitus', 'diabetes_mellitus_with_end_organ_damage',
              'chronic_kidney_disease', 'hemiplegia_from_stroke', 'leukemia', 'malignant_lymphoma', 'localized_solid_tumor',
              'metastatic_solid_tumor', 'mild_liver_disease', 'moderate_to_severe_liver_disease', 'aids'] + label_cols]
    
    # Fill missing values with np.nan
    df = df.fillna(value=np.nan)
             
    if processed_file_name is not None:
        
        # Save csv file with processed data
        df.to_csv(processed_file_name, index=None)
        
    return df


def beautify_column_names (df):
    '''Renames columns' names of KNIGHT clinical dataframe (clinical features' names).'''

    df.rename(columns={"age_at_nephrectomy": "Age at nephrectomy",
                        "body_mass_index": "Body mass index (BMI)",
                        "smoking_level":"Smoking level",
                        "chewing_tobacco_use": "Chewing tobacco use",
                        "alcohol_use":"Alcohol use",
                        "radiographic_size":"Radiographic size",
                        "gender":"Gender",
                        "myocardial_infarction":"Has myocardial infarction",
                        "congestive_heart_failure": "Has congestive heart failure",
                        "peripheral_vascular_disease":"Has peripheral vascular disease",
                        "cerebrovascular_disease":"Has cerebrovascular disease",
                        "dementia":"Has dementia",
                        "copd":"Has chronic obstructive pulmonary disease (COPD)",
                        "connective_tissue_disease":"Has connective tissue disease",
                        "peptic_ulcer_disease":"Has peptic ulcer disease",
                        "uncomplicated_diabetes_mellitus":"Has uncomplicated diabetes mellitus",
                        "diabetes_mellitus_with_end_organ_damage":"Has diabetes mellitus with end organ demage",
                        "chronic_kidney_disease":"Has chronic kidney disease",
                        "hemiplegia_from_stroke":"Has hemiglegia from stroke",
                        "leukemia":"Has leukemia",
                        "malignant_lymphoma":"Has malignant lymphoma",
                        "localized_solid_tumor":"Has localized solid tumor",
                        "metastatic_solid_tumor":"Has metastatic solid tumor",
                        "mild_liver_disease": "Has mild liver disease",
                        "moderate_to_severe_liver_disease":"Has moderate to severe liver disease",
                        "aids":"Has AIDS",
                        "last_preop_egfr_value": "Preoperative eGFR value (ml/min)",
                        "days_before_nephrectomy_egfr_was_measured":"Days before nephrectomy at which eGFR was measured",
                       "age_when_quit_smoking":"Age when quit smoking",
                       "ever_smoked": "Has smoking history",

                        }, inplace=True)
    return df


def get_KNIGHT_data_split(knight_data: pd.DataFrame, train_path = 'KNIGHT/knight/data/knight_train_set.pkl',
                                                 val_path = 'KNIGHT/knight/data/knight_val_set.pkl',
                                                 splits_file_path = 'splits_final.pkl',
                                                 ):
    """
    Splits Knight dataframe into train and validation sets based on the case ids. 
    Args:
    knight_data: knight processed dataframe to split.
    train_path (str): path to save the train set.
    val_path (str): path to save the validation set.
    Returns (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,): matrix (x) and labels (y) for train and val sets, respectively 
    """
    
        
    if os.path.isfile(splits_file_path):
        splits = pkl.load(open(splits_file_path, 'rb')) 

    # For this example, we use split 0 out of the 5 available cross-validation splits
        split = splits[0]

        train_set = knight_data.loc[split['train']]
        x_train = train_set.drop(columns=[x for x in train_set.columns if 'label' in x])
        y_train = train_set[[x for x in train_set.columns if 'label' in x]]

        val_set = knight_data.loc[split['val']]
        x_val = val_set.drop(columns=[x for x in val_set.columns if 'label' in x])
        y_val = val_set[[x for x in val_set.columns if 'label' in x]]                                 


        pkl.dump((x_train, y_train), open(train_path, 'wb')) 
        pkl.dump((x_val, y_val), open(val_path, 'wb')) 

        print('Number of patients in train: %d and val: %d' %(x_train.shape[0], x_val.shape[0]))

    return x_train, y_train, x_val, y_val
