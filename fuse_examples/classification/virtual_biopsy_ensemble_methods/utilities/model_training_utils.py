import xgboost as xgb
import pickle as pkl
import pandas as pd
import numpy as np
import compare_auc_delong
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

import shap
from pathlib import Path


# Warning supressed for now: 'ntree_limit is deprecated, use `iteration_range` or model slicing instead.'
import warnings
warnings.filterwarnings('ignore')



TRAINED_MODELS_DIR = Path(__file__).resolve().parent.parent / "trained_models_output"
SHAPLEY_VALUES_DIR = Path(__file__).resolve().parent.parent / "shapley_values"
PREDICTIONS_DIR = Path(__file__).resolve().parent.parent / "predicted_probabilities_output"


########################### XGBOOST HYPERPARAMETER GRID SEARCH ########################### 

MIN_CHILD_WEIGHT = [1, 2, 3, 5, 7]  
GAMMA= [0, 0.1, 0.2, 0.3, 0.4]
COLSAMPLE_BYTREE = [0.3, 0.4, 0.5, 0.7, 1.0]
LEARNING_RATE= [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]  
MAX_DEPTH= [3, 4, 5, 6, 7, 8, 10] 

GRID_PARAMS = {'min_child_weight': MIN_CHILD_WEIGHT,
                 'gamma': GAMMA,
                 'colsample_bytree': COLSAMPLE_BYTREE,
                 'learning_rate': LEARNING_RATE,
                 'max_depth': MAX_DEPTH}

########################### ########################### ########################### 


def get_destination(directory, feature_set, risk_class, model_index, create):
    
    if directory == PREDICTIONS_DIR:
        
        destination = directory /'{}_predicted_probs_{}.csv'.format(feature_set, risk_class)
        
    else:
        
        destination = directory /'{}_{}_model_run_{:02d}.pkl'.format(feature_set, risk_class, model_index)
    
    if create and not destination.parent.exists():
        destination.parent.mkdir()
        
    return destination

def train_model(x_train, x_val, y_train, y_val, feature_set, n_iter, risk_class):
    '''feature_set = subset_clinical_and_imaging_feats, all_clinical_feats, subset_clinical_feats, imaging
       risk_class = risk_label_0, risk_label_1, risk_label_2, risk_label_3, risk_label_4'''
    
    
    ########## Define feature sets
    
    

    model = xgb.XGBClassifier (eval_metric = 'auc')

    model_CV = RandomizedSearchCV(estimator = model,
                             param_distributions = GRID_PARAMS, 
                              scoring = 'roc_auc', 
                              n_iter = 30, cv=5) 
    
    if feature_set == 'all_clinical_feats':
        
        x_train = x_train
        x_val = x_val
        
    elif feature_set == 'subset_clinical_feats':
        
        x_train = x_train [subset_clinical_feats]
        x_val = x_val [subset_clinical_feats]
        
    elif feature_set == 'imaging':
        
        x_train = x_train [imaging_feats]
        x_val = x_val [imaging_feats]
    
    else: 
        
        x_train = x_train [shap_feats + imaging_feats] 
        x_val = x_val [shap_feats + imaging_feats] 
        
    stats_runs = {}
    stats_runs['models_predictions_' + str(risk_class)] = []    
    
    print('**Training models for class {} **'.format(risk_class))

    for i in range(n_iter):
   
        print ('{}/{}\r'.format(i+1, n_iter), end = '', flush=True)
        print('\n')

        model_CV.fit(x_train, y_train[risk_class])
        y_pred = model_CV.predict(x_val)
        prob = model_CV.predict_proba(x_val)[:,1]

        stats_runs['models_predictions_' + str(risk_class)].append(prob)
        
        print('\tAUC of model {}: {:.2f} [{:.2f}, {:.2f}]'.format(i+1, roc_auc_score(y_val[risk_class], 
                                        prob), *compare_auc_delong.get_delong_ci(prob, y_val[risk_class])))

        
        ###### SAVE THE BEST MODEL ########
            
        dst_pkls = get_destination(directory = TRAINED_MODELS_DIR, feature_set = feature_set, 
                              risk_class = risk_class, model_index = i, create = True)
        pkl.dump(model_CV.best_estimator_, open(dst_pkls, 'wb'))
            

        # If all clinical features are used, do SHAP analysis to find a subset of most contributing clinical features
        
        if feature_set == 'all_clinical_feats':

            # Get parameters of the best search
            best_params_grid_search = model_CV.best_params_
            best_model  = xgb.XGBClassifier(**best_params_grid_search)
            
            best_model.fit(x_train, y_train[risk_class])
            
            # Get shapley values with TreeExplainer
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(x_val)
            
            # Save shapley values as pickle file
            
            dst_shapley = get_destination(directory = SHAPLEY_VALUES_DIR, feature_set = feature_set, 
                              risk_class = risk_class, model_index = i, create = True)
            pkl.dump(shap_values, open(dst_shapley, 'wb'))
            

    # Save models predictions
    dst_probs = get_destination(directory = PREDICTIONS_DIR, feature_set = feature_set, 
                              risk_class = risk_class, model_index = None, create = True)
    np.savetxt(dst_probs, stats_runs['models_predictions_' + str(risk_class)], delimiter=',')
    
    
def plot_AUC_validation_set(ax, feature_set, risk_class):
    ''''''

    # Read trained models predictions on validation set
    dst_probs = get_destination(directory = PREDICTIONS_DIR, feature_set = feature_set, 
                              risk_class = risk_class, model_index = None, create = False)
    
    model_probabilities = pd.read_csv(dst_probs, header=None)
    
    # Read ground truth labels
    label = pd.read_pickle('KNIGHT/knight/data/knight_val_set.pkl')[1][risk_class]
    
    # Compute AUC for all runs
    fpr = []
    tpr = []

    for i in range(model_probabilities.shape[0]):

        fpr.append(roc_curve(label.values, model_probabilities.iloc[i, :])[0])
        tpr.append(roc_curve(label.values, model_probabilities.iloc[i, :])[1])

    # Get FPR and TPR of ensemble model
    fpr_ensemble, tpr_ensemble, _ = roc_curve(label.values, model_probabilities.mean())
    roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)
    
    
    fontsize=13
    lw = 1
    
    # Plot thick line for AUC of emsemble model
    ax.plot(fpr_ensemble, tpr_ensemble, color = 'darkorange', lw=lw+2, label = str(risk_class) + ': AUC = %0.2f' %roc_auc_ensemble),

    # Plot shaded lines for AUCs of 10 runs
    for i in range(model_probabilities.shape[0]):
        
        ax.plot(fpr[i], tpr[i], color = 'darkorange', alpha=0.2, lw=lw, zorder=1)

    # Plot diagonal line
    ax.plot([0,1], [0,1], color='black', lw=lw, linestyle = '--') 
    ax.set_ylabel('Sensitivity', fontsize=fontsize)
    ax.set_xlabel('1 - Specificity', fontsize=fontsize)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5), fontsize=fontsize)
    ax.set_yticks(np.linspace(0, 1, 5), fontsize=fontsize)
    ax.legend( loc = 'lower right', prop = {'size': 14})
    ax.axis('equal')
    
    
def show_shap_summary_plot (feature_set, risk_class, model_index, x_val ):
    '''feature set needs to be all clinical feats'''
    
    
    dst_shap = get_destination(directory = SHAPLEY_VALUES_DIR, feature_set = feature_set, 
                              risk_class = risk_class, model_index = model_index, create = False)
    
    shapley_vals = pkl.load(open(dst_shap, 'rb'))
    
    return shap.summary_plot(shapley_vals, x_val, max_display = 10, plot_type='dot')



def get_feature_importance (classes, n_iter, feature_set, n_top_features, x_val):
    '''feature set needs to be all clinical feats'''

    # zip shapley values of all runs into a dictionary
    
    shapley_values = {}
    
    shapley_values['risk_label_0'] = []
    shapley_values['risk_label_1'] = []
    shapley_values['risk_label_2'] = []
    shapley_values['risk_label_3'] = []
    shapley_values['risk_label_4'] = []
    
    for risk_class in classes:
        for model_index in range(n_iter):
            
            
            dst_shap = get_destination(directory = SHAPLEY_VALUES_DIR, feature_set = feature_set, 
                              risk_class = risk_class, model_index = model_index, create = False)
                        
            shapley_values[risk_class].append(pkl.load(open(dst_shap, 'rb')))
          
    
    # For each run, get the name of clinical features ordered by importance and zip into another dict
    
    top_n_features = {}
    
    top_n_features['risk_label_0'] = []
    top_n_features['risk_label_1'] = []
    top_n_features['risk_label_2'] = []
    top_n_features['risk_label_3'] = []
    top_n_features['risk_label_4'] = []
    
    for risk_class in classes:
        for model_index in range(n_iter):
            top_n_features[risk_class].append(x_val.columns[np.argsort(np.abs(shapley_values[risk_class][model_index]).mean(0))].tolist()[::-1][:n_top_features])
        
    # Flatten all lists

    top_n_features['risk_label_0'] = [item for sublist in top_n_features['risk_label_0']  for item in sublist]
    top_n_features['risk_label_1'] = [item for sublist in top_n_features['risk_label_1']  for item in sublist]
    top_n_features['risk_label_2'] = [item for sublist in top_n_features['risk_label_2']  for item in sublist]
    top_n_features['risk_label_3'] = [item for sublist in top_n_features['risk_label_3']  for item in sublist]
    top_n_features['risk_label_4'] = [item for sublist in top_n_features['risk_label_4']  for item in sublist]
    
   # We only select the features that appeared at least 5 times out of the ten runs for each class  

    top_risk_label_0 = set([i for i in top_n_features['risk_label_0'] if top_n_features['risk_label_0'].count(i) > 5])
    top_risk_label_1 = set([i for i in top_n_features['risk_label_1'] if top_n_features['risk_label_1'].count(i) > 5])
    top_risk_label_2 = set([i for i in top_n_features['risk_label_2'] if top_n_features['risk_label_2'].count(i) > 5])
    top_risk_label_3 = set([i for i in top_n_features['risk_label_3'] if top_n_features['risk_label_3'].count(i) > 5])
    top_risk_label_4 = set([i for i in top_n_features['risk_label_4'] if top_n_features['risk_label_4'].count(i) > 5])
        
    subset_clinical_feats = list(set().union(top_risk_label_0, top_risk_label_1, top_risk_label_2, top_risk_label_3, top_risk_label_4))
    
    print('Most contributing features to the prediction of all classes are:\n')
    print(', '.join(subset_feats))
      
    return subset_clinical_feats

        