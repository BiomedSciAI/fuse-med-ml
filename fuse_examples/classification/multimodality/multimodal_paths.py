import os

def multimodal_paths(dataset_name,root_data,root, experiment,cache_path):
    if dataset_name=='mg_clinical':
        paths = {
            # paths
            'data_dir': root_data,
            'tabular_filename': os.path.join(root_data, 'mg_clinical_dicom_sentra/fx_sentara_cohort_processed.csv'),
            'imaging_filename': os.path.join(root_data, 'mg_clinical_dicom_sentra/mg_sentara_cohort.csv'),
            'train_val_test_filenames': [os.path.join(root_data, 'mg_clinical_dicom_sentra/sentara_train_pathologies.csv'),
                                         os.path.join(root_data, 'mg_clinical_dicom_sentra/sentara_val_pathologies.csv'),
                                         os.path.join(root_data, 'mg_clinical_dicom_sentra/sentara_test_pathologies.csv'), ],

            # keys to extract from dataframe
            'key_columns': ['patient_id', 'study_id'],
            'sample_key': 'sample_desc',
            'label_key': 'finding_biopsy',
            'img_key': 'dcm_url',

            'model_dir': os.path.join(root, experiment, 'model_mg_clinical_dicom_sentra'),
            'force_reset_model_dir': True,
            # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
            'cache_dir': os.path.join(cache_path, '/lala/'),
            'inference_dir': os.path.join(root, experiment, 'infer_mg_clinical_dicom_sentra')}
    if dataset_name == 'mg_radiologic':
        paths = {
    # paths
    'data_dir': root_data,
    'tabular_filename': os.path.join(root_data, 'mg_radiologist_usa/dataset_MG_clinical.csv'),
    'imaging_filename': os.path.join(root_data, 'mg_radiologist_usa/mg_usa_cohort.csv'),
    'train_val_test_filenames': [os.path.join(root_data, 'mg_radiologist_usa/dataset_MG_clinical_train.csv'),
                                 os.path.join(root_data, 'mg_radiologist_usa/dataset_MG_clinical_validation.csv'),
                                 os.path.join(root_data, 'mg_radiologist_usa/dataset_MG_clinical_heldout.csv'), ],

    # keys to extract from dataframe
    'key_columns': ['patient_id'],
    'sample_key': 'sample_desc',
    'label_key': 'finding_biopsy',
    'img_key': 'dcm_url',

    'model_dir': os.path.join(root_data,'model_mg_radiologist_usa/'+experiment),
    'force_reset_model_dir': True,
    # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
    'cache_dir': os.path.join(cache_path),
    'inference_dir': os.path.join(root_data,'model_mg_radiologist_usa/'+experiment)}


    return paths