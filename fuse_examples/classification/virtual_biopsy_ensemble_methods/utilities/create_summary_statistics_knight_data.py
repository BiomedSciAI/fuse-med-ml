import pandas as pd

def format_table_1_stats(df, feature_name, is_binary, to_int=False):
    """
    Formats each cell of table 1 according to feature type
    """
    if is_binary:
        return '{:.0f} ({:.1f})'.format(df[feature_name].sum(),
                                         (df[feature_name].sum() * 100 / df.shape[0]))
    # continuous features
    if to_int:
        str_format = '{:.0f} \u00B1 {:.0f}'
    else:
        str_format = '{:.1f} \u00B1 {:.1f}'
    return str_format.format(*df[feature_name].agg(['mean', 'std']))


def print_table_1_features(name_df_tuples):
    """
    Prints the table part containing the features (and outcomes title)
    """
    # basic structure
    df_table = pd.DataFrame(
        index=['No. of patients', 
               'Number of women', 
               'Age at nephrectomy (y)', 
               'Most recent body mass index', 
               'Preoperative eGFR value (ml/min)',
               'Smoking history',
               'Myocardial infarction history',               
               'Has chronic kidney disease',
#                'Has congestive heart failure',
#                'Has peripheral vascular disease',
#                'Has cerebrovascular disease',
#                'Has connective tissue disease',
#                'Has peptic ulcer disease',
#                'Has chronic obstructive pulmonary disease (COPD)',
#                'Has mild liver disease',
#                'Has moderate to severe liver disease',
#                'Postoperative outcomes', 
#               'Days before nephrectomy at which eGFR was measured',             
              'Outcome/Risk group', 
              'Adjuvant therapy candidacy',   
              ])

    for name, df in name_df_tuples:
        df_table[name] = [
            df.shape[0],  # num patients
            format_table_1_stats(df, 'Gender', True),  # women
            format_table_1_stats(df, 'Age at nephrectomy', False, True), 
            format_table_1_stats(df, 'Body mass index (BMI)', False),  # BMI
            format_table_1_stats(df, 'Preoperative eGFR value (ml/min)', False), 
            format_table_1_stats(df, 'Has smoking history', True),
            format_table_1_stats(df, 'Has myocardial infarction', True),            
            format_table_1_stats(df, 'Has chronic kidney disease', True),

#             format_table_1_stats(df, 'Has congestive heart failure', True),
#             format_table_1_stats(df, 'Has peripheral vascular disease', True),
#             format_table_1_stats(df, 'Has cerebrovascular disease', True),
#             format_table_1_stats(df, 'Has connective tissue disease', True),
#             format_table_1_stats(df, 'Has peptic ulcer disease', True),
#             format_table_1_stats(df, 'Has chronic obstructive pulmonary disease (COPD)', True),
#             format_table_1_stats(df, 'Has mild liver disease', True),
#             format_table_1_stats(df, 'Has moderate to severe liver disease', True),
#             format_table_1_stats(df, 'Days before nephrectomy at which eGFR was measured', False),
            '', 
            format_table_1_stats(df, 'adj_therapy_label', True),
            ]  

    return df_table


def print_table_1_outcomes(name_df_tuples): 
                                   
    """
    Prints the table part containing the outcomes stats
    """
    # basic structure
    df_table = pd.DataFrame(index=['Benign', 'Low risk', 'Intermediate risk',
                                   'High risk', 'Very high risk'])

    for name, df in name_df_tuples:
        # count outcome events
        num_per_outcome =  df['risk_label'].value_counts(dropna=False)

        # sort index 
        num_per_outcome.sort_index(inplace=True)
        # add a stats column per data set
        df_table[name] = ['{:.0f} ({:.1f})'.format(x, x * 100 / df.shape[0]) for x in num_per_outcome]
        
    return df_table

def print_table_1(name_df_tuples):
    return pd.concat([print_table_1_features(name_df_tuples), print_table_1_outcomes(name_df_tuples)])
