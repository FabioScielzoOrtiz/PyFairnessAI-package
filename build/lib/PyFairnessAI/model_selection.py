import numpy as np
import pandas as pd
import time
import random
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from PyFairnessAI.metrics import (statistical_parity_difference, abs_statistical_parity_difference, disparate_impact_ratio,
                                  abs_equal_opportunity_difference, average_odds_error,
                                  false_positive_rate_difference, false_negative_rate_difference, true_positive_rate_difference,
                                  true_negative_rate_difference, false_positive_rate_ratio, false_negative_rate_ratio,
                                  true_positive_rate_ratio, true_negative_rate_ratio, positive_predicted_value_difference,
                                  positive_predicted_value_ratio, positive_predicted_value_abs_difference) 

###############################################################################################################################

fairness_metrics = {'statistical_parity_difference': statistical_parity_difference,
                    'abs_statistical_parity_difference': abs_statistical_parity_difference,
                    'disparate_impact_ratio': disparate_impact_ratio,
                    'abs_equal_opportunity_difference': abs_equal_opportunity_difference,
                    'average_odds_error': average_odds_error,
                    'false_positive_rate_difference': false_positive_rate_difference,
                    'false_negative_rate_difference': false_negative_rate_difference,
                    'true_positive_rate_difference': true_positive_rate_difference,
                    'true_negative_rate_difference': true_negative_rate_difference,
                    'false_positive_rate_ratio': false_positive_rate_ratio,
                    'false_negative_rate_ratio': false_negative_rate_ratio,
                    'true_positive_rate_ratio': true_positive_rate_ratio,
                    'true_negative_rate_ratio': true_negative_rate_ratio,
                    'positive_predicted_value_difference': positive_predicted_value_difference,
                    'positive_predicted_value_ratio': positive_predicted_value_ratio,
                    'positive_predicted_value_abs_difference': positive_predicted_value_abs_difference
                    }


def cross_val_score_fairness(estimator, X, y, prot_attr, priv_group, pos_label, scoring, cv):
    
    # X must be a Pandas DataFrame (in order to read the prot_attr, and preserve its indexation based on prot_attr, what is need for fairness post-processors)
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a Pandas DataFrame')
    # y must be a Pandas Series (in order to preserve its indexation based on prot_attr, what is need for fairness post-processors)
    if not isinstance(y, pd.Series):
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        else:
            TypeError('y must be a Pandas Series')

    metric_iters = []
    # Split the data into training and validation sets 
    for train_index, val_index in cv.split(X, y): 
        try:
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            Y_train, Y_val = y.iloc[train_index], y.iloc[val_index]
            A_val = X_val[prot_attr] # sensitive variable in val set

            # Training the estimator
            estimator.fit(X_train, Y_train)

            # Predicting on validation set
            Y_val_hat = estimator.predict(X_val)

            # Calculate fairness metrics for each iteration of the cross-validation process
            metric_iters.append(fairness_metrics[scoring](y_true=Y_val, y_pred=Y_val_hat, prot_attr=A_val,
                                                          priv_group=priv_group, pos_label=pos_label))
        except Exception as e:
            print('cross_val_score_fairness Exception:', e)
            
    # Compute the average of the metric along the iterations
    final_metric = np.mean(metric_iters)

    return final_metric, metric_iters

###############################################################################################################################

def combined_score(predictive_scores, fairness_scores, 
                   predictive_scoring_direction, 
                   fairness_scoring_direction,
                   predictive_weight, fairness_weight):

    scaler = MinMaxScaler(feature_range=(0,1)) 
    predictive_scores_normalized = scaler.fit_transform(np.array(predictive_scores).reshape(-1, 1)).flatten()
    fairness_scores_normalized = scaler.fit_transform(np.array(fairness_scores).reshape(-1, 1)).flatten()
    if predictive_scoring_direction == 'minimize':
        predictive_scores_normalized = 1 - predictive_scores_normalized
    if fairness_scoring_direction == 'minimize':
        fairness_scores_normalized = 1 - fairness_scores_normalized
    if  predictive_weight + fairness_weight == 1:
        combined_scores = predictive_scores_normalized * predictive_weight + fairness_scores_normalized * fairness_weight
    else:
        raise ValueError("The sum of predictive_weight and fairness_weight must be 1.")
    
    return combined_scores
    
###############################################################################################################################

class RandomizedSearchCVFairness:
    
    def __init__(self, estimator, param_distributions, 
                 fairness_scoring, predictive_scoring, objective, 
                 fairness_scoring_direction, predictive_scoring_direction, 
                 fairness_weight, predictive_weight, 
                 cv, n_iter, random_state, prot_attr, priv_group, pos_label):

        self.estimator = estimator
        self.param_distributions = param_distributions
        self.fairness_scoring = fairness_scoring
        self.predictive_scoring = predictive_scoring
        self.objective = objective # ['fairness', 'predictive']
        self.fairness_scoring_direction = fairness_scoring_direction # ['maximize', ' minimize']
        self.predictive_scoring_direction = predictive_scoring_direction # ['maximize', ' minimize']
        self.fairness_weight = fairness_weight
        self.predictive_weight = predictive_weight
        self.cv = cv 
        self.n_iter = n_iter
        self.random_state = random_state
        self.prot_attr = prot_attr
        self.priv_group = priv_group
        self.pos_label = pos_label
        self.results_ = []

    def _random_param_sample(self):
        """Randomly sample a parameter combination from the distributions."""
        random_params = {key: random.choice(val) for key, val in self.param_distributions.items()}
        return random_params
    
    def fit(self, X, y):
        
        random.seed(self.random_state)
        for iter in range(self.n_iter):
            print(f'Iteration {iter}\nStarted')
            start_time = time.time()  
            try:
                random_params = self._random_param_sample()
                self.estimator.set_params(**random_params)

                fairness_final_metric, _ = cross_val_score_fairness(estimator=self.estimator, X=X, y=y, 
                                                                    prot_attr=self.prot_attr, 
                                                                    priv_group=self.priv_group,
                                                                    pos_label=self.pos_label, 
                                                                    scoring=self.fairness_scoring, cv=self.cv)  
                predictive_metric_iters = cross_val_score(estimator=self.estimator, X=X, y=y, 
                                                          scoring=self.predictive_scoring, cv=self.cv)
                
                predictive_final_metric = np.mean(predictive_metric_iters)
                 
                self.results_.append({'params': random_params, 
                                      'predictive-score': predictive_final_metric, 
                                      'fairness-score': fairness_final_metric})
                
            except Exception as e:
                print('RandomizedSearchCVFairness Exception:', e)

            end_time = time.time()            
            print(f'Finished\nTime: {np.round(end_time - start_time, 2)} secs\n-----------------------------------')
            
        predictive_scores = [self.results_[i][f'predictive-score'] for i in range(len(self.results_))]
        fairness_scores = [self.results_[i][f'fairness-score'] for i in range(len(self.results_))]
        
        # Computing the combined score
        combined_scores = combined_score(predictive_scores=predictive_scores, 
                                         fairness_scores=fairness_scores, 
                                         predictive_scoring_direction=self.predictive_scoring_direction, 
                                         fairness_scoring_direction=self.fairness_scoring_direction,
                                         predictive_weight=self.predictive_weight, 
                                         fairness_weight=self.fairness_weight)
       
        for i in range(len(self.results_)):
            self.results_[i]['combined-score'] = combined_scores[i]
        
        # Optimizing the parameters according to the objective and the scores
        # Obtaining the best params and score, and building a data-frame with the results
        score_list = [self.results_[i][f'{self.objective}-score'] for i in range(len(self.results_))]
        self.cv_results_ = pd.DataFrame(self.results_)

        scoring_direction_map = {'combined': ('maximize', False),
                                    'fairness': (self.fairness_scoring_direction, None),
                                    'predictive': (self.predictive_scoring_direction, None)
                                }
        scoring_direction, ascending_value = scoring_direction_map[self.objective]
        opt_function = np.argmax if scoring_direction == 'maximize' else np.argmin
        ascending_value = False if scoring_direction == 'maximize' else True 
        best_score_idx = opt_function(score_list)
        self.cv_results_ = self.cv_results_.sort_values(by=f'{self.objective}-score', ascending=ascending_value)
        self.best_params_ = self.results_[best_score_idx]['params']
        self.best_score_ = self.results_[best_score_idx][f'{self.objective}-score']
