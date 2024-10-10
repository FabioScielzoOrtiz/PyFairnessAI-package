import numpy as np
import pandas as pd

from aif360.sklearn.datasets import fetch_german

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import tensorflow as tf

from PyMachineLearning.preprocessing import Encoder, Scaler, Imputer, ColumnTransformerToPandas
from PyMachineLearning.models import LogisticRegressionThreshold

from PyFairnessAI.model_selection import RandomizedSearchCVFairness, combined_score
from PyFairnessAI.preprocessing import ReweighingMetaEstimator
from PyFairnessAI.inprocessing import (AdversarialDebiasingEstimator, 
                                       ExponentiatedGradientReductionEstimator, 
                                       GridSearchReductionEstimator, Moment)
from PyFairnessAI.postprocessing import CalibratedEqualizedOdds, RejectOptionClassifier, PostProcessingMeta

import time 

import warnings
warnings.filterwarnings("ignore")

#########################################################################################################

X, y = fetch_german(binary_age=True)
response_favorable_label = 1 # 'good' before encoding
sens_variable = 'age' # name of sensitive variable
X[sens_variable] = X.apply(lambda row: 1 if row[sens_variable] >= 25 else 0, axis=1).astype('category')
sens_priv_group = 1 # >= 25 years
A = X[sens_variable] # sensitive variable / protected attribute
quant_predictors = [col for col in X.columns if X.dtypes[col] != 'category']
cat_predictors = [col for col in X.columns if col not in quant_predictors]  
predictors = quant_predictors + cat_predictors # X.columns

encoder = Encoder(method='ordinal')
y = pd.Series(encoder.fit_transform(y.to_numpy().reshape(-1, 1)).flatten())

# needed for fairness post-processing
X.index = X[sens_variable] 
y.index = X[sens_variable]  

#########################################################################################################

random_state = 123
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=random_state, stratify=y)
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

#########################################################################################################

# Set up TensorFlow session (required by AdversarialDebiasingEstimator)
tf_session = tf.compat.v1.Session
# disable_eager_execution is required as well by TensorFlow
tf.compat.v1.disable_eager_execution()
tf.random.set_seed(random_state)

#########################################################################################################

quant_pipeline = Pipeline([
    ('imputer', Imputer()),
    ('scaler', Scaler())
    ])

cat_pipeline = Pipeline([
    ('imputer', Imputer(method='simple_most_frequent')),
    ('encoder', Encoder(method='one-hot', drop='first'))
    ])

quant_cat_processing = ColumnTransformer(transformers=[('quant', quant_pipeline, quant_predictors),
                                                       ('cat', cat_pipeline, cat_predictors)])

#########################################################################################################

models, pipelines = {}, {}

ceo = CalibratedEqualizedOdds(prot_attr=sens_variable, random_state=random_state) # Fairnes post-processor
roc = RejectOptionClassifier(prot_attr=sens_variable) # Fairnes post-processor

models['log_reg'] = LogisticRegressionThreshold(solver='liblinear', random_state=random_state)
models['XGB'] = XGBClassifier(random_state=random_state)
models['LGB'] = LGBMClassifier(random_state=random_state)
models['RF'] = RandomForestClassifier(random_state=random_state)

models['log_reg_reweighing'] = ReweighingMetaEstimator(estimator=models['log_reg'], prot_attr=sens_variable) # Fairness Pre-Processor
models['XGB_reweighing'] = ReweighingMetaEstimator(estimator=models['XGB'], prot_attr=sens_variable) # Fairness Pre-Processor
models['LGB_reweighing'] = ReweighingMetaEstimator(estimator=models['LGB'], prot_attr=sens_variable) # Fairness Pre-Processor
models['RF_reweighing'] = ReweighingMetaEstimator(estimator=models['RF'], prot_attr=sens_variable) # Fairness Pre-Processor

models['adv_debiasing'] = AdversarialDebiasingEstimator(prot_attr=sens_variable, scope_name='classifier', random_state=random_state) # Fairness In-Processor

models['log_reg_expGR'] = ExponentiatedGradientReductionEstimator(prot_attr=sens_variable, estimator=models['log_reg']) # Fairness In-Processor
models['XGB_expGR'] = ExponentiatedGradientReductionEstimator(prot_attr=sens_variable, estimator=models['XGB']) # Fairness In-Processor
models['LGB_expGR'] = ExponentiatedGradientReductionEstimator(prot_attr=sens_variable, estimator=models['LGB']) # Fairness In-Processor
models['RF_expGR'] = ExponentiatedGradientReductionEstimator(prot_attr=sens_variable, estimator=models['RF']) # Fairness In-Processor
models['log_reg_GSR'] = GridSearchReductionEstimator(prot_attr=sens_variable, estimator=models['log_reg']) # Fairness In-Processor
models['XGB_GSR'] = GridSearchReductionEstimator(prot_attr=sens_variable, estimator=models['XGB']) # Fairness In-Processor
models['LGB_GSR'] = GridSearchReductionEstimator(prot_attr=sens_variable, estimator=models['LGB']) # Fairness In-Processor
models['RF_GSR'] = GridSearchReductionEstimator(prot_attr=sens_variable, estimator=models['RF']) # Fairness In-Processor

models['log_reg_CEO'] = PostProcessingMeta(estimator=models['log_reg'], postprocessor=ceo, prefit=False, val_size=0.25) # Fairnes post-processor
models['XGB_CEO'] = PostProcessingMeta(estimator=models['XGB'], postprocessor=ceo, prefit=False, val_size=0.25) # Fairnes post-processor
models['LGB_CEO'] = PostProcessingMeta(estimator=models['LGB'], postprocessor=ceo, prefit=False, val_size=0.25) # Fairnes post-processor
models['RF_CEO'] = PostProcessingMeta(estimator=models['RF'], postprocessor=ceo, prefit=False, val_size=0.25) # Fairnes post-processor

models['log_reg_ROC'] = PostProcessingMeta(estimator=models['log_reg'], postprocessor=roc, prefit=False, val_size=0.25) # Fairnes post-processor
models['XGB_ROC'] = PostProcessingMeta(estimator=models['XGB'], postprocessor=roc, prefit=False, val_size=0.25) # Fairnes post-processor
models['LGB_ROC'] = PostProcessingMeta(estimator=models['LGB'], postprocessor=roc, prefit=False, val_size=0.25) # Fairnes post-processor
models['RF_ROC'] = PostProcessingMeta(estimator=models['RF'], postprocessor=roc, prefit=False, val_size=0.25) # Fairnes post-processor

models['log_reg_reweighing_expGR'] = ReweighingMetaEstimator(estimator=models['log_reg_expGR'], prot_attr=sens_variable) # pre + in
models['XGB_reweighing_expGR'] = ReweighingMetaEstimator(estimator=models['XGB_expGR'], prot_attr=sens_variable) # pre + in
models['LGB_reweighing_expGR'] = ReweighingMetaEstimator(estimator=models['LGB_expGR'], prot_attr=sens_variable) # pre + in
models['RF_reweighing_expGR'] = ReweighingMetaEstimator(estimator=models['RF_expGR'], prot_attr=sens_variable) # pre + in

models['log_reg_reweighing_GSR'] = ReweighingMetaEstimator(estimator=models['log_reg_GSR'], prot_attr=sens_variable) # pre + in
models['XGB_reweighing_GSR'] = ReweighingMetaEstimator(estimator=models['XGB_GSR'], prot_attr=sens_variable) # pre + in
models['LGB_reweighing_GSR'] = ReweighingMetaEstimator(estimator=models['LGB_GSR'], prot_attr=sens_variable) # pre + in
models['RF_reweighing_GSR'] = ReweighingMetaEstimator(estimator=models['RF_GSR'], prot_attr=sens_variable) # pre + in

models['log_reg_reweighing_CEO'] = ReweighingMetaEstimator(estimator=models['log_reg_CEO'], prot_attr=sens_variable) # pre + post
models['XGB_reweighing_CEO'] = ReweighingMetaEstimator(estimator=models['XGB_CEO'], prot_attr=sens_variable) # pre + post
models['LGB_reweighing_CEO'] = ReweighingMetaEstimator(estimator=models['LGB_CEO'], prot_attr=sens_variable) # pre + post
models['RF_reweighing_CEO'] = ReweighingMetaEstimator(estimator=models['RF_CEO'], prot_attr=sens_variable) # pre + post

models['log_reg_reweighing_ROC'] = ReweighingMetaEstimator(estimator=models['log_reg_ROC'], prot_attr=sens_variable) # pre + post
models['XGB_reweighing_ROC'] = ReweighingMetaEstimator(estimator=models['XGB_ROC'], prot_attr=sens_variable) # pre + post
models['LGB_reweighing_ROC'] = ReweighingMetaEstimator(estimator=models['LGB_ROC'], prot_attr=sens_variable) # pre + post
models['RF_reweighing_ROC'] = ReweighingMetaEstimator(estimator=models['RF_ROC'], prot_attr=sens_variable) # pre + post

models['log_reg_CEO_reweighing'] = PostProcessingMeta(estimator=models['log_reg_reweighing'], postprocessor=ceo, prefit=False, val_size=0.25)  # post + pre
models['XGB_CEO_reweighing'] = PostProcessingMeta(estimator=models['XGB_reweighing'], postprocessor=ceo, prefit=False, val_size=0.25)  # post + pre
models['LGB_CEO_reweighing'] = PostProcessingMeta(estimator=models['LGB_reweighing'], postprocessor=ceo, prefit=False, val_size=0.25)  # post + pre
models['RF_CEO_reweighing'] = PostProcessingMeta(estimator=models['RF_reweighing'], postprocessor=ceo, prefit=False, val_size=0.25)  # post + pre

models['log_reg_ROC_reweighing'] = PostProcessingMeta(estimator=models['log_reg_reweighing'], postprocessor=roc, prefit=False, val_size=0.25)  # post + pre
models['XGB_ROC_reweighing'] = PostProcessingMeta(estimator=models['XGB_reweighing'], postprocessor=roc, prefit=False, val_size=0.25)  # post + pre
models['LGB_ROC_reweighing'] = PostProcessingMeta(estimator=models['LGB_reweighing'], postprocessor=roc, prefit=False, val_size=0.25)  # post + pre
models['RF_ROC_reweighing'] = PostProcessingMeta(estimator=models['RF_reweighing'], postprocessor=roc, prefit=False, val_size=0.25)  # post + pre

models['log_reg_CEO_expGR'] = PostProcessingMeta(estimator=models['log_reg_expGR'], postprocessor=ceo, prefit=False, val_size=0.25)  # post + in
models['XGB_CEO_expGR'] = PostProcessingMeta(estimator=models['XGB_expGR'], postprocessor=ceo, prefit=False, val_size=0.25)  # post + in
models['LGB_CEO_expGR'] = PostProcessingMeta(estimator=models['LGB_expGR'], postprocessor=ceo, prefit=False, val_size=0.25)  # post + in
models['RF_CEO_expGR'] = PostProcessingMeta(estimator=models['RF_expGR'], postprocessor=ceo, prefit=False, val_size=0.25)  # post + in

models['log_reg_ROC_expGR'] = PostProcessingMeta(estimator=models['log_reg_expGR'], postprocessor=roc, prefit=False, val_size=0.25)  # post + in
models['XGB_ROC_expGR'] = PostProcessingMeta(estimator=models['XGB_expGR'], postprocessor=roc, prefit=False, val_size=0.25)  # post + in
models['LGB_ROC_expGR'] = PostProcessingMeta(estimator=models['LGB_expGR'], postprocessor=roc, prefit=False, val_size=0.25)  # post + in
models['RF_ROC_expGR'] = PostProcessingMeta(estimator=models['RF_expGR'], postprocessor=roc, prefit=False, val_size=0.25)  # post + in

models['log_reg_CEO_GSR'] = PostProcessingMeta(estimator=models['log_reg_GSR'], postprocessor=ceo, prefit=False, val_size=0.25)  # post + in
models['XGB_CEO_GSR'] = PostProcessingMeta(estimator=models['XGB_GSR'], postprocessor=ceo, prefit=False, val_size=0.25)  # post + in
models['LGB_CEO_GSR'] = PostProcessingMeta(estimator=models['LGB_GSR'], postprocessor=ceo, prefit=False, val_size=0.25)  # post + in
models['RF_CEO_GSR'] = PostProcessingMeta(estimator=models['RF_GSR'], postprocessor=ceo, prefit=False, val_size=0.25)  # post + in

models['log_reg_ROC_GSR'] = PostProcessingMeta(estimator=models['log_reg_GSR'], postprocessor=roc, prefit=False, val_size=0.25)  # post + in
models['XGB_ROC_GSR'] = PostProcessingMeta(estimator=models['XGB_GSR'], postprocessor=roc, prefit=False, val_size=0.25)  # post + in
models['LGB_ROC_GSR'] = PostProcessingMeta(estimator=models['LGB_GSR'], postprocessor=roc, prefit=False, val_size=0.25)  # post + in
models['RF_ROC_GSR'] = PostProcessingMeta(estimator=models['RF_GSR'], postprocessor=roc, prefit=False, val_size=0.25)  # post + in

#########################################################################################################

fairness_pre_processors_names = ['reweighing']
fairness_in_processors_names = ['expGR', 'GSR', 'adv_debiasing'] 
fairness_post_processors_names = ['CEO', 'ROC']
fairness_pre_in_processors_names = ['reweighing_expGR', 'reweighing_GSR']
fairness_pre_post_processors_names = ['reweighing_CEO', 'reweighing_ROC']
fairness_post_pre_processors_names = ['CEO_reweighing', 'ROC_reweighing']
fairness_post_in_processors_names = ['CEO_expGR', 'ROC_expGR', 'CEO_GSR', 'ROC_GSR']
fairness_multi_processors_names = fairness_pre_in_processors_names + fairness_pre_post_processors_names + fairness_post_pre_processors_names + fairness_post_in_processors_names
fairness_processors_names = fairness_pre_processors_names + fairness_in_processors_names + fairness_post_processors_names + fairness_multi_processors_names

#########################################################################################################

for key, model in models.items():

    if  any(x in key for x in fairness_processors_names): # model[key] involves fairness processor

        pipelines[key] = Pipeline([
                # Fairness processors need a Pandas df X as input to read the sens_variable_name
                ('preprocessing', ColumnTransformerToPandas(column_transformer=quant_cat_processing,
                                                            prot_attr=sens_variable,  
                                                            prot_attr_index=True)), # prot_attr_index=True needed for fairness post-processors
                (key, model) 
                ])            

    else:

        pipelines[key] = Pipeline([
                ('preprocessing', quant_cat_processing),
                (key, model) 
                ])

#########################################################################################################
 
preprocessing_grid = {}

preprocessing_grid['not_fairness_processor'] = {'preprocessing__quant__scaler__apply': [True, False],
                                                'preprocessing__quant__scaler__method': ['standard', 'min-max'],
                                                'preprocessing__cat__encoder__method': ['ordinal', 'one-hot'],
                                                'preprocessing__cat__imputer__apply':  [False],
                                                'preprocessing__quant__imputer__apply': [False],
                                               #'preprocessing__cat__imputer__method':  ['simple_most_frequent'],
                                               #'preprocessing__quant__imputer__method': ['simple_mean', 'simple_median']
                                                }

# Same as  preprocessing_grid['not_fairness_preprocessor'] but adding '__column_transformer__' to the keys (needed in ColumnTransformerToPandas)        
preprocessing_grid['fairness_processor'] = {'__'.join([k.split('__')[0]] + ['column_transformer'] + k.split('__')[1:]) : v 
                                            for k, v in preprocessing_grid['not_fairness_processor'].items()}

#########################################################################################################

def get_key_preprocessing_grid(model):

    if any(x in model for x in fairness_multi_processors_names): # Fairness multi processor involved in the model
        key1 = model + '__estimator' + '__estimator'
        key3 = None
        preprocessing_grid_ = preprocessing_grid['fairness_processor'].copy()
        if any(x in model for x in fairness_pre_in_processors_names):
            key2 = model + '__estimator'
        elif any(x in model for x in fairness_pre_post_processors_names):
            key2 = model + '__estimator' + '__postprocessor'
        elif any(x in model for x in fairness_post_pre_processors_names):
            key2 = model + '__postprocessor'   
        elif any(x in model for x in fairness_post_in_processors_names):
           key2 = model + '__estimator'
           key3 = model + '__postprocessor'   
    else:
        key2 = key3 = None
        if any(x in model for x in fairness_pre_processors_names + fairness_in_processors_names): # Fairness pre or in processor involved in the model
            if model == 'adv_debiasing':
                key1 = model
            else:
                key1 = model + '__estimator' 
            preprocessing_grid_ = preprocessing_grid['fairness_processor'].copy()
        elif any(x in model for x in fairness_post_processors_names): # Fairness post processor involved in the model
            key1 = model + '__estimator' 
            key2 = model + '__postprocessor'
            preprocessing_grid_ = preprocessing_grid['fairness_processor'].copy()       
        else: # No fairness processor involved
            key1 = model
            preprocessing_grid_ = preprocessing_grid['not_fairness_processor'].copy()
    
    return key1, key2, key3, preprocessing_grid_

#########################################################################################################

def get_model_param_grid(model, key):

    if 'log_reg' in model:

        param_grid = {f'{key}__penalty': ['l1', 'l2'],
                      f'{key}__C':  [0.01, 0.1, 1, 10, 30, 50, 75, 100],
                      f'{key}__class_weight': ['balanced', None],
                      f'{key}__threshold': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    }
    
    elif 'XGB' in model:

        param_grid = {f'{key}__max_depth': [10, 20, 30, 40, 50, 70, 100],
                      f'{key}__reg_lambda': np.arange(0, 1, 0.05),
                      f'{key}__n_estimators':  [30, 50, 70, 100],
                      f'{key}__eta': np.arange(0, 0.3, 0.02),
                      f'{key}__alpha': np.arange(0.2, 1, 0.01)
                    }

    elif 'LGB' in model:

        param_grid = {f'{key}__max_depth': np.arange(2, 50),
                      f'{key}__num_leaves':  np.arange(2, 50),
                      f'{key}__n_estimators': [30, 50, 70, 100, 120, 150],
                      f'{key}__learning_rate':  np.arange(0.001, 0.1, 0.003),
                      f'{key}__lambda_l1': np.arange(0.001, 1, 0.005),
                      f'{key}__lambda_l2': np.arange(0.001, 1, 0.005),
                      f'{key}__min_split_gain':  np.arange(0.001, 0.01, 0.001),
                      f'{key}__min_child_weight': np.arange(5, 50),
                      f'{key}__lambda_feature_fraction':  np.arange(0.1, 0.95, 0.05)                                                                                                           
                      }
    
    elif 'RF' in model:
        
        param_grid = {f'{key}__max_depth': np.arange(2, 15),
                      f'{key}__min_samples_leaf':  np.arange(2, 15),
                      f'{key}__min_samples_split':  np.arange(2, 15),
                      f'{key}__n_estimators': [30, 50, 70, 100, 120],
                      f'{key}__criterion': ['gini', 'entropy']                                                                                                 
                    }
        
    elif 'adv_debiasing' in model:
        
        param_grid = {f'{key}__adversary_loss_weight': np.arange(0.01, 1, 0.03),
                      f'{key}__num_epochs':  np.arange(10, 100),
                      f'{key}__batch_size':  np.arange(70, 200),
                      f'{key}__classifier_num_hidden_units': np.arange(70, 300),
                      f'{key}__debias': [True, False]   
                    }
        
    return param_grid

#########################################################################################################

def get_processor_param_grid(processor, key):

    if processor == 'expGR':

        param_grid = {f'{key}__constraints': ['DemographicParity', 'EqualizedOdds', 
                                        'TruePositiveRateParity', 'ErrorRateParity'],
                      f'{key}__eps': np.arange(0.001, 0.1, 0.003),
                      f'{key}__max_iter': np.arange(20, 100, 5),
                      f'{key}__eta0': np.arange(0.1, 4, 0.2),
                      f'{key}__drop_prot_attr': [True, False],
                    }   
        
    elif processor == 'GSR':

        param_grid = {f'{key}__constraints': ['DemographicParity', 'EqualizedOdds', 
                                            'TruePositiveRateParity', 'ErrorRateParity'],
                     f'{key}__constraint_weight': np.arange(0.01, 1, 0.05),
                     f'{key}__grid_size': np.arange(5, 50, 5),
                     f'{key}__grid_limit': np.arange(1, 10),
                     f'{key}__loss': ['ZeroOne', 'Square', 'Absolute'],                                
                     f'{key}__drop_prot_attr': [True, False],
                    }  
    
    elif processor == 'CEO':

        param_grid = {f'{key}__cost_constraint': ['fpr', 'fnr', 'weighted']}

    elif processor == 'ROC':

        param_grid = {f'{key}__threshold': np.arange(0.05, 0.5, 0.03), # must be between 0-0.5
                      f'{key}__margin': np.arange(0.01, 0.05, 0.005) # must be between 0-0.05
                    }
        
    return param_grid

#########################################################################################################

def get_pipeline_param_grid(model):

    key1, key2, key3, preprocessing_grid_ = get_key_preprocessing_grid(model)
    param_grid = preprocessing_grid_.copy()
    param_grid.update(get_model_param_grid(model, key=key1)) 
    
    if any(x in model for x in fairness_multi_processors_names): # Multi processor involved

        if 'reweighing_' in model: 

            param_grid.update(get_processor_param_grid(processor=model.split('_')[-1], key=key2)) 
        
        elif '_reweighing' in model: 

            param_grid.update(get_processor_param_grid(processor=model.split('_')[-2], key=key2)) 

        else:

            param_grid.update(get_processor_param_grid(processor=model.split('_')[-1], key=key2))
            param_grid.update(get_processor_param_grid(processor=model.split('_')[-2], key=key3))

    else: # Not multi processor (no fairness processor or pre/in/post processor)
                
        if any(x in model for x in fairness_in_processors_names): # in processor  
            if model != 'adv_debiasing':
                param_grid.update(get_processor_param_grid(processor=model.split('_')[-1], key=model))

        elif any(x in model for x in fairness_post_processors_names): # post processor 

            param_grid.update(get_processor_param_grid(processor=model.split('_')[-1], key=key2))
    
    return param_grid

def n_iter(model):

    if 'XGB' in model:
        return 5
    elif 'RF' in model:
        return 10
    else:
        return 20

#########################################################################################################

param_grid, best_results_list = {}, []

for model in pipelines.keys():

    print(model)

    param_grid[model] = get_pipeline_param_grid(model)

    fairness_random_search = RandomizedSearchCVFairness(estimator=pipelines[model], 
                                                        param_distributions=param_grid[model], 
                                                        fairness_scoring='average_odds_error', 
                                                        predictive_scoring='balanced_accuracy',
                                                        objective='combined', 
                                                        fairness_scoring_direction='minimize',
                                                        predictive_scoring_direction='maximize',
                                                        fairness_weight=0.5, predictive_weight=0.5,
                                                        cv=inner, n_iter=n_iter(model), random_state=random_state,
                                                        prot_attr=sens_variable, 
                                                        priv_group=sens_priv_group,
                                                        pos_label=response_favorable_label)

    start_time = time.time()
    fairness_random_search.fit(X=X_train, y=y_train)
    end_time = time.time()
    elapsed_time = np.round(end_time - start_time, 3)
    best_result_dict = {'model': model, 'time': elapsed_time}
    best_result_dict.update(dict(fairness_random_search.cv_results_.iloc[0]))
    best_results_list.append(best_result_dict)

#########################################################################################################

best_results = pd.DataFrame(best_results_list)
best_results['combined-score'] = combined_score(predictive_scores=best_results['predictive-score'], 
                                               fairness_scores=best_results['fairness-score'], 
                                               predictive_scoring_direction='maximize', 
                                               fairness_scoring_direction='minimize',
                                               predictive_weight=0.5, fairness_weight=0.5)
best_results = best_results.sort_values(by='combined-score', ascending=False)


best_results.to_csv(r'notebooks\Project-notebooks\results\best_results_fairness_workflow.csv')
print('Results saved as CSV')