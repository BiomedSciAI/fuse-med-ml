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

from typing import Optional
from pydantic import BaseSettings
from pydantic import Field
from typing import Any
import ehrtransformers.configs.naming as naming


class DataSettings(BaseSettings):
    days_to_inddate: Optional[int] = 180 #days before index date when the latest visit is considered (later visits are discarded)
    days_to_inddate_start: Optional[
        int] = 180  # days before index date when the first visit is considered (earlier visits are discarded)
    event_prediction_window_days: Optional[int] = None #within what window to predict an event
    treatment_event_prediction_window_days: Optional[int] = None #within what window to predict treatment event

    days_to_inddate_tr: Optional[int] = 180  #days before index date when the latest visit is considered (later visits are discarded) in the train set (in case we want to do it differently for train and test)
    days_to_inddate_start_tr: Optional[
        int] = 180  # days before index date when the earliest visit is considered (earlier visits are discarded) in the train set (in case we want to do it differently for train and test)
    event_prediction_window_days_tr: Optional[int] = None
    limit_visits: Optional[int] = None
    out_type: Optional[str] = "LABEL"
    task: Optional[str] = "PD"
    subtask: Optional[str] = "MEDICARE_INPATIENT"
    task_type: Optional[str] = "outcome"
    output_name: Optional[str] = "admdate_visit"
    use_procedures: Optional[bool] = False #Whether or not to use procedure codes in addition to diagnosis codes when constructing an input
    num_loader_workers: Optional[int] = 3
    visit_days_resolution: Optional[int] = 2  # consecutive visits that are less than visit_days_resolution days apart are merged into one visit.
    # Note that merging stops only when number of days between consecutive visits is more than visit_days_resolution, so
    # within a group of merged visits there may be some more than visit_days_resolution days apart.
    # E.g. for visit days 1, 3, 5, 9, and for visit_days_resolution=2, days 1, 3, and 5 will be merged into a single visit, and day 10 will be a separate visit
    input_column_names: Optional[list] = None
    min_visits_per_patient: Optional[int] = 5  # patients with less visits are ignored

    # When predicting next visit, this will map regular visit dictionary (codes to indices, index per code) to reduced
    # visit dictionary (codes to indices, code groups mapped to (fewer) indices). When no mapping is used, this is null.
    reduce_pred_visit_vocab_mapping_path: Optional[str] = "/ehrtransformers/data_access/icd_to_ccs.json"  # null
    data_source: Optional[str] = "csv"  # sql_db #csv
    data_source_str: Optional[str] = "/data/usr/vadim/EHR/rc_for_ehr_sample.csv" #full data source descriptor (sql code or path)
    sample_data_source_str: Optional[str] = "/data/usr/vadim/EHR/rc_for_ehr_sample.csv"  # sample data source descriptor (sql code or path)
    gt_source_str: Optional[str] = None

class OptimizationSettings(BaseSettings):
    lr: Optional[float] = 3.0e-06  # 1e-4, #3e-6,
    warmup_proportion: Optional[float] = 0.1
    weight_decay: Optional[float] = 0.01


class LearningSettings(BaseSettings):
    batch_size: Optional[int] = 384  # 64
    device: Optional[str] = "cuda:0"
    is_data_parallel: Optional[bool] = False #If we use multiple GPUs, this parallellizes the application of the model across them (torch.nn.DataParallel)
    gradient_accumulation_steps: Optional[int] = 1
    save_model: Optional[bool] = True
    stop_file: Optional[str] = "stop.txt"
    optimization: Optional[OptimizationSettings] = Field(
        default_factory=OptimizationSettings
    )


class GlobalSettings(BaseSettings):
    age_symbol: Optional[str] = None
    debug_mode: Optional[bool] = False #runs in debug mode, which means that the output is directed to stdout, and not to a log file
    global_stat_file: Optional[str] = "global_stats.csv"
    max_age: Optional[int] = 110
    max_len_seq: Optional[int] = 100
    max_label_len: Optional[int] = 7
    min_visit: Optional[int] = 5
    # choice whether to use months for age (=1) or years (=12)
    month: Optional[int] = 1
    # path utils to all EHR experiments
    uber_base_path: Optional[str] = "/data/usr/vadim/EHR/"


class ModelSettings(BaseSettings):
    # number of vocab for age embedding
    age_vocab_size: Optional[int] = None
    # multi-head attention dropout rate
    attention_probs_dropout_prob: Optional[float] = 0.22
    # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    hidden_act: Optional[str] = "gelu"
    # dropout rate
    hidden_dropout_prob: Optional[float] = 0.2
    # word embedding and seg embedding hidden size (needs to be a multiple of attention heads)
    hidden_size: Optional[int] = 240  # 288,
    # parameter weight initializer range
    initializer_range: Optional[float] = 0.02
    # the size of the "intermediate" layer in the transformer encoder
    intermediate_size: Optional[int] = 512
    # maximum number of tokens
    max_position_embedding: Optional[int] = 100
    # number of attention heads
    num_attention_heads: Optional[int] = 12
    # number of multi-head attention layers required
    num_hidden_layers: Optional[int] = 6
    # MLM pretrained model path
    # cf "/data/usr/vadim/EHR/PD_MEDICARE_INPATIENT_180_to_ind/LABEL/models/PD/MLM_small_test"
    pretrain_model: Optional[str] = None
    seg_vocab_size: Optional[int] = None  # number of vocab for seg embedding
    train_epochs: Optional[int] = 50
    vocab_size: Optional[int] = None  # number of disease + symbols for word embedding
    reverse_input_direction: Optional[bool] = False #Whether or not to reverse the input direction (i.e. first visit becomes the last visit the model sees)
    heads: Optional[list] = None
    head_weights: Optional[list] = None
    sampler_weights: Optional[list] = None
    contrastive_instances: Optional[int] = 2  # number of patient visits used in contrastive loss (reducing distance between embeddings of the same patient)


class NamingConventionSettings(BaseSettings):
    age_key: Optional[str] = "AGE"
    age_month_key: Optional[str] = "AGE_MON"
    date_key: Optional[str] = "ADMDATE"
    index_date_key: Optional[str] = "INDDATE" #index date of the reference disease

    adm_date_key: Optional[str] = "ADMDATE" # admission date - this and service date are for internal use of the db extractor only. TODO: remove them from here
    svc_date_key: Optional[str] = "SVCDATE" # service date - this and admission date are for internal use of the db extractor only. TODO: remove them from here

    date_birth_key: Optional[str] = "DOBYR" #name of a column containing birth year
    diagnosis_vec_key: Optional[str] = "DX" #name of a column containing input codes (diagnoses, perscriptions, etc.)
    gender_key: Optional[str] = "SEX"   #name of a column containing gender information. TODO: separate this to input DB and internal DB versions
    male_val: Optional[str] = '1'       #value in gender_key column signifying that the patient is male
    female_val: Optional[str] = '2'     #value in gender_key column signifying that the patient is female
    label_key: Optional[str] = "label"
    outcome_key: Optional[str] = "PD"
    disease_key: Optional[str] = None #not used
    event_key: Optional[str] = "EVENTS"
    treatment_event_key: Optional[str] = "TREATMENT_EVENTS"
    patient_id_key: Optional[str] = "ENROLID"
    separator_str: Optional[str] = naming.separator_token
    dxver_key: Optional[str] = "DXVER"
    healthy_val: Optional[Any] = 0
    sick_val: Optional[Any] = 1
    split_key: Optional[str] = "split" #column that contains split assignment (strings 'train', 'validation' or 'test')
    fold_key: Optional[str] = "fold" #column that contains fold assignment (ints 0..n-1)
    next_visit_key: Optional[str] = "DXNEXTVIS"


class EHRTransformerConfigSettings(BaseSettings):
    data: Optional[DataSettings] = Field(default_factory=DataSettings)
    learning: Optional[LearningSettings] = Field(default_factory=LearningSettings)
    global_settings: Optional[GlobalSettings] = Field(alias="global", default_factory=GlobalSettings)    
    model: Optional[ModelSettings] = Field(default_factory=ModelSettings)
    naming_conventions: Optional[NamingConventionSettings] = Field(default_factory=NamingConventionSettings)
