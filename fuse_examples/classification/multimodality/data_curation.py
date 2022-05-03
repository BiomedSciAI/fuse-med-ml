from MedicalAnalyticsCore.DatabaseUtils.selected_studies_queries import get_annotations_and_findings
from MedicalAnalyticsCore.DatabaseUtils.tableResolver import TableResolver
from MedicalAnalyticsCore.DatabaseUtils.connection import create_homer_engine, Connection
from MedicalAnalyticsCore.DatabaseUtils import tableNames
from MedicalAnalyticsCore.DatabaseUtils import db_utils as db


def mg_data_curation(imaging_filename):
    REVISION_DATE = '20200915'
    TableResolver().set_revision(REVISION_DATE)
    revision = {'prefix': 'sentara', 'suffix': REVISION_DATE}
    engine = Connection().get_engine()

    df_with_findings = get_annotations_and_findings(engine, revision,
                                                    exam_types=['MG'], viewpoints=None,  # ['CC','MLO'], \
                                                    include_findings=True, remove_invalids=True,
                                                    remove_heldout=False, \
                                                    remove_excluded=False, remove_less_than_4views=False, \
                                                    load_from_file=False, save_to_file=False)

    my_providers = ['sentara']
    df = df_with_findings.loc[df_with_findings['provider'].isin(my_providers)]
    # fixing assymetry
    asymmetries = ['asymmetry', 'developing asymmetry', 'focal asymmetry', 'global asymmetry']
    df['is_asymmetry'] = df['pathology'].isin(asymmetries)
    df['is_Breast_Assymetry'] = df['type'].isin(['Breast Assymetry'])
    df.loc[df['is_asymmetry'], 'pathology'] = df[df['is_asymmetry']]['biopsy_outcome']
    df.loc[df['is_Breast_Assymetry'], 'pathology'] = df[df['is_Breast_Assymetry']]['biopsy_outcome']
    # remove duble xmls
    aa_unsorted = df
    aa_unsorted.sort_values('xml_url', ascending=False, inplace=True)
    xml_url_to_keep = aa_unsorted.groupby(['image_id'])['xml_url'].transform('first')
    df = aa_unsorted[aa_unsorted['xml_url'] == xml_url_to_keep]
    remove_from_pathology = ['undefined', 'not_applicable', 'Undefined', 'extracapsular rupture of breast implant',
                             'intracapsular rupture of breast implant']
    is_pathology = ~df.pathology.isnull() & ~df.pathology.isin(remove_from_pathology)
    is_digital = df.image_source == 'Digital'
    is_biopsy = df.finding_biopsy.isin(['negative', 'negative high risk', 'positive'])
    df = df[(is_digital) & (is_pathology) & (is_biopsy)]
    df.to_csv(imaging_filename)

#------------------Baseline
def apply_gluon_baseline(train_set,test_set,label,save_path):
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(label=label, path=save_path, eval_metric='roc_auc').fit(train_set)
    results = predictor.fit_summary(show_plot=True)

    # Inference time:
    y_test = test_set[label]
    test_data = test_set.drop(labels=[label],
                               axis=1)  # delete labels from test data since we wouldn't have them in practice
    print(test_data.head())

    predictor = TabularPredictor.load(
        save_path)
    y_pred = predictor.predict_proba(test_data)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)