import numpy as np
import pandas as pd

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


def cross_val_score_fairness(estimator, X, y, sens_variable_name, priv_group, pos_label, scoring, cv):
    
    # X must be a Pandas DataFrame
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a Pandas DataFrame')
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    metric_iters = []
    # Split the data into training and validation sets 
    for train_index, val_index in cv.split(X, y): 
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        Y_train, Y_val = y[train_index], y[val_index]
        A_val = X_val[sens_variable_name] # sensitive variable in val set

        # Train logistic regression model
        estimator.fit(X_train, Y_train)

        # Predict on validation set
        Y_val_hat = estimator.predict(X_val)

        # Calculate fairness metrics for each iteration of the cross-validation process
        metric_iters.append(fairness_metrics[scoring](y_true=Y_val, y_pred=Y_val_hat, prot_attr=A_val,
                                                      priv_group=priv_group, pos_label=pos_label))
            
    # Compute the average of the metric along the iterations
    final_metric = np.mean(metric_iters)

    return final_metric, metric_iters