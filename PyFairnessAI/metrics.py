###############################################################################################################

import numpy as np
import pandas as pd
from aif360.sklearn.metrics import (statistical_parity_difference, disparate_impact_ratio, 
                                    equal_opportunity_difference, average_odds_error)
epsilon = 1e-6

#############################################################################################################################################

def abs_statistical_parity_difference(y_true, y_pred, prot_attr, priv_group, pos_label):

    value = statistical_parity_difference(y_true=y_true, y_pred=y_pred, prot_attr=prot_attr, priv_group=priv_group, pos_label=pos_label)
    
    return np.abs(value)

#############################################################################################################################################

def abs_equal_opportunity_difference(y_true, y_pred, prot_attr, priv_group, pos_label):

    value = equal_opportunity_difference(y_true=y_true, y_pred=y_pred, prot_attr=prot_attr, priv_group=priv_group, pos_label=pos_label)
  
    return np.abs(value)

#############################################################################################################################################

def check_data_type(y_true, prot_attr):

    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(prot_attr, pd.Series):
        prot_attr = prot_attr.to_numpy()  

    return y_true, prot_attr

def get_neg_label(y_true, pos_label):

    unique_y_true = np.unique(y_true)
    if len(unique_y_true) == 1:
        raise ValueError("y_true contains only one unique value, unable to determine negative label.")
    else:
        neg_label_list = [x for x in unique_y_true if x != pos_label]
        neg_label = neg_label_list[0] if len(unique_y_true) == 2 else neg_label_list
    
    return neg_label

def get_unpriv_group(prot_attr, priv_group):

    unique_prot_attr = np.unique(prot_attr)
    if len(unique_prot_attr) == 1:
        raise ValueError("prot_attr contains only one unique value, unable to determine unprivileged group.")  
    else:
        unpriv_group_list = [x for x in unique_prot_attr if x != priv_group]
        unpriv_group = unpriv_group_list[0] if len(unique_prot_attr) == 2 else unpriv_group_list

    return unpriv_group

def false_positive_rate_privileged(y_true, y_pred, prot_attr, priv_group, pos_label):

    y_true, prot_attr = check_data_type(y_true, prot_attr)
    neg_label = get_neg_label(y_true, pos_label)

    true_negative_idx = np.where(y_true == neg_label)[0]  if len(np.unique(y_true)) == 2 else np.where(y_true in neg_label)[0] 
    predicted_positive_idx = np.where(y_pred == pos_label)[0]  
    priv_idx = np.where(prot_attr == priv_group)[0] 
    true_negative_priv_idx = np.intersect1d(true_negative_idx, priv_idx)
    true_negative_priv_pred_pos_idx = np.intersect1d(true_negative_priv_idx, predicted_positive_idx)
    FPR_priv_den = len(true_negative_priv_pred_pos_idx)
    FPR_priv_num = len(true_negative_priv_idx)
    FPR_priv =  FPR_priv_den / FPR_priv_num if FPR_priv_num != 0 else FPR_priv_den / epsilon

    return FPR_priv 

def false_positive_rate_unprivileged(y_true, y_pred, prot_attr, priv_group, pos_label):

    y_true, prot_attr = check_data_type(y_true, prot_attr)
    neg_label = get_neg_label(y_true, pos_label)
    unpriv_group = get_unpriv_group(prot_attr, priv_group)

    true_negative_idx = np.where(y_true == neg_label)[0]  if len(np.unique(y_true)) == 2 else np.where(y_true in neg_label)[0] 
    predicted_positive_idx = np.where(y_pred == pos_label)[0]  
    unpriv_idx = np.where(prot_attr == unpriv_group)[0] if len(np.unique(prot_attr)) == 2 else np.where(prot_attr in unpriv_group)[0]
    true_negative_unpriv_idx = np.intersect1d(true_negative_idx, unpriv_idx)
    true_negative_unpriv_pred_pos_idx = np.intersect1d(true_negative_unpriv_idx, predicted_positive_idx)
    FPR_unpriv_den = len(true_negative_unpriv_pred_pos_idx) 
    FPR_unpriv_num = len(true_negative_unpriv_idx)
    FPR_unpriv = FPR_unpriv_den / FPR_unpriv_num if FPR_unpriv_num != 0 else FPR_unpriv_den / epsilon

    return FPR_unpriv 

def false_negative_rate_privileged(y_true, y_pred, prot_attr, priv_group, pos_label):

    y_true, prot_attr = check_data_type(y_true, prot_attr)
    neg_label = get_neg_label(y_true, pos_label)
       
    true_positive_idx = np.where(y_true == pos_label)[0]    
    predicted_negative_idx = np.where(y_pred == neg_label)[0] if len(np.unique(y_true)) == 2 else  np.where(y_true in neg_label)[0]     
    priv_idx = np.where(prot_attr == priv_group)[0] if len(np.unique(prot_attr)) == 2 else np.where(prot_attr in priv_group)[0]
    true_positive_priv_idx = np.intersect1d(true_positive_idx, priv_idx)
    true_positive_priv_pred_neg_idx = np.intersect1d(true_positive_priv_idx, predicted_negative_idx)
    FNR_priv_num = len(true_positive_priv_pred_neg_idx)
    FNR_priv_den = len(true_positive_priv_idx)
    FNR_priv =  FNR_priv_num / FNR_priv_den if FNR_priv_den != 0 else FNR_priv_num / epsilon

    return FNR_priv

def false_negative_rate_unprivileged(y_true, y_pred, prot_attr, priv_group, pos_label):

    y_true, prot_attr = check_data_type(y_true, prot_attr)
    neg_label = get_neg_label(y_true, pos_label)
    unpriv_group = get_unpriv_group(prot_attr, priv_group)
       
    true_positive_idx = np.where(y_true == pos_label)[0]    
    predicted_negative_idx = np.where(y_pred == neg_label)[0] if len(np.unique(y_true)) == 2 else  np.where(y_true in neg_label)[0]     
    unpriv_idx = np.where(prot_attr == unpriv_group)[0] if len(np.unique(prot_attr)) == 2 else np.where(prot_attr in unpriv_group)[0]
    true_positive_unpriv_idx = np.intersect1d(true_positive_idx, unpriv_idx)
    true_positive_unpriv_pred_neg_idx = np.intersect1d(true_positive_unpriv_idx, predicted_negative_idx)
    FNR_unpriv_num = len(true_positive_unpriv_pred_neg_idx)
    FNR_unpriv_den = len(true_positive_unpriv_idx)
    FNR_unpriv = FNR_unpriv_num / FNR_unpriv_den if FNR_unpriv_den != 0 else FNR_unpriv_num / epsilon

    return FNR_unpriv

def true_negative_rate_privileged(y_true, y_pred, prot_attr, priv_group, pos_label):
    
    y_true, prot_attr = check_data_type(y_true, prot_attr)
    neg_label = get_neg_label(y_true, pos_label)
    
    if len(np.unique(y_true)) == 2:
        true_negative_idx = np.where(y_true == neg_label)[0]   
        predicted_negative_idx = np.where(y_pred == neg_label)[0]  
    else:
        true_negative_idx = np.where(y_true in neg_label)[0] 
        predicted_negative_idx = np.where(y_pred in neg_label)[0] 
    priv_idx = np.where(prot_attr == priv_group)[0] 
    true_negative_priv_idx = np.intersect1d(true_negative_idx, priv_idx)
    true_negative_priv_pred_neg_idx = np.intersect1d(true_negative_priv_idx, predicted_negative_idx)
    TNR_priv_num = len(true_negative_priv_pred_neg_idx)
    TNR_priv_den = len(true_negative_priv_idx)
    TNR_priv = TNR_priv_num / TNR_priv_den if TNR_priv_den != 0 else TNR_priv_num / epsilon

    return TNR_priv

def true_negative_rate_unprivileged(y_true, y_pred, prot_attr, priv_group, pos_label):
    
    y_true, prot_attr = check_data_type(y_true, prot_attr)
    neg_label = get_neg_label(y_true, pos_label)
    unpriv_group = get_unpriv_group(prot_attr, priv_group)

    if len(np.unique(y_true)) == 2:
        true_negative_idx = np.where(y_true == neg_label)[0]   
        predicted_negative_idx = np.where(y_pred == neg_label)[0]  
    else:
        true_negative_idx = np.where(y_true in neg_label)[0] 
        predicted_negative_idx = np.where(y_pred in neg_label)[0] 
    unpriv_idx = np.where(prot_attr == unpriv_group)[0] if len(np.unique(prot_attr)) == 2 else np.where(prot_attr in unpriv_group)[0]
    true_negative_unpriv_idx = np.intersect1d(true_negative_idx, unpriv_idx)
    true_negative_unpriv_pred_neg_idx = np.intersect1d(true_negative_unpriv_idx, predicted_negative_idx)
    TNR_unpriv_num = len(true_negative_unpriv_pred_neg_idx)
    TNR_unpriv_den = len(true_negative_unpriv_idx)
    TNR_unpriv = TNR_unpriv_num / TNR_unpriv_den if TNR_unpriv_den != 0 else TNR_unpriv_num / epsilon

    return TNR_unpriv

def true_positive_rate_privileged(y_true, y_pred, prot_attr, priv_group, pos_label):

    y_true, prot_attr = check_data_type(y_true, prot_attr)
    
    true_positive_idx = np.where(y_true == pos_label)[0]  
    predicted_positive_idx = np.where(y_pred == pos_label)[0]  
    priv_idx = np.where(prot_attr == priv_group)[0] 
    true_positive_priv_idx = np.intersect1d(true_positive_idx, priv_idx)
    true_positive_priv_pred_pos_idx = np.intersect1d(true_positive_priv_idx, predicted_positive_idx)
    TPR_priv_num = len(true_positive_priv_pred_pos_idx)
    TPR_priv_den = len(true_positive_priv_idx) 
    TPR_priv = TPR_priv_num / TPR_priv_den if TPR_priv_den != 0 else TPR_priv_num / epsilon   

    return TPR_priv

def true_positive_rate_unprivileged(y_true, y_pred, prot_attr, priv_group, pos_label):

    y_true, prot_attr = check_data_type(y_true, prot_attr)
    unpriv_group = get_unpriv_group(prot_attr, priv_group)
    
    true_positive_idx = np.where(y_true == pos_label)[0]  
    predicted_positive_idx = np.where(y_pred == pos_label)[0]  
    unpriv_idx = np.where(prot_attr == unpriv_group)[0] if len(np.unique(prot_attr)) == 2 else np.where(prot_attr in unpriv_group)[0]
    true_positive_unpriv_idx = np.intersect1d(true_positive_idx, unpriv_idx)
    true_positive_unpriv_pred_pos_idx = np.intersect1d(true_positive_unpriv_idx, predicted_positive_idx)
    TPR_unpriv_num = len(true_positive_unpriv_pred_pos_idx)
    TPR_unpriv_den = len(true_positive_unpriv_idx)
    TPR_unpriv = TPR_unpriv_num / TPR_unpriv_den if TPR_unpriv_den != 0 else TPR_unpriv_num / epsilon

    return TPR_unpriv

def false_positive_rate_difference(y_true, y_pred, prot_attr, priv_group, pos_label):

    FPR_unpriv = false_positive_rate_unprivileged(y_true, y_pred, prot_attr, priv_group, pos_label)
    FPR_priv = false_positive_rate_privileged(y_true, y_pred, prot_attr, priv_group, pos_label)
    FPR_diff = FPR_unpriv - FPR_priv

    return FPR_diff

def false_negative_rate_difference(y_true, y_pred, prot_attr, priv_group, pos_label):

    FNR_unpriv = false_negative_rate_unprivileged(y_true, y_pred, prot_attr, priv_group, pos_label)
    FNR_priv = false_negative_rate_privileged(y_true, y_pred, prot_attr, priv_group, pos_label)
    FNR_diff = FNR_unpriv - FNR_priv  

    return FNR_diff

def true_positive_rate_difference(y_true, y_pred, prot_attr, priv_group, pos_label):

    TPR_diff = - false_negative_rate_difference(y_true, y_pred, prot_attr, priv_group, pos_label)

    return TPR_diff

def true_negative_rate_difference(y_true, y_pred, prot_attr, priv_group, pos_label):

    TNR_diff = - false_positive_rate_difference(y_true, y_pred, prot_attr, priv_group, pos_label)

    return TNR_diff

def false_positive_rate_ratio(y_true, y_pred, prot_attr, priv_group, pos_label):

    FPR_unpriv = false_positive_rate_unprivileged(y_true, y_pred, prot_attr, priv_group, pos_label)
    FPR_priv = false_positive_rate_privileged(y_true, y_pred, prot_attr, priv_group, pos_label)
    FPR_ratio = FPR_unpriv / FPR_priv if FPR_priv != 0 else FPR_unpriv / epsilon

    return FPR_ratio

def false_negative_rate_ratio(y_true, y_pred, prot_attr, priv_group, pos_label):

    FNR_unpriv = false_negative_rate_unprivileged(y_true, y_pred, prot_attr, priv_group, pos_label)
    FNR_priv = false_negative_rate_privileged(y_true, y_pred, prot_attr, priv_group, pos_label)
    FNR_ratio = FNR_unpriv / FNR_priv if FNR_priv != 0 else FNR_unpriv / epsilon

    return FNR_ratio

def true_positive_rate_ratio(y_true, y_pred, prot_attr, priv_group, pos_label):

    TPR_unpriv = true_positive_rate_unprivileged(y_true, y_pred, prot_attr, priv_group, pos_label)
    TPR_priv = true_positive_rate_privileged(y_true, y_pred, prot_attr, priv_group, pos_label)
    TPR_ratio = TPR_unpriv / TPR_priv if TPR_priv != 0 else TPR_unpriv / epsilon

    return TPR_ratio

def true_negative_rate_ratio(y_true, y_pred, prot_attr, priv_group, pos_label):

    TNR_unpriv = true_negative_rate_unprivileged(y_true, y_pred, prot_attr, priv_group, pos_label)
    TNR_priv = true_negative_rate_privileged(y_true, y_pred, prot_attr, priv_group, pos_label)
    TNR_ratio = TNR_unpriv / TNR_priv if TNR_priv != 0 else TNR_unpriv / epsilon

    return TNR_ratio

#############################################################################################################################################

def positive_predicted_value_unpriv(y_true, y_pred, prot_attr, priv_group, pos_label):

    y_true, prot_attr = check_data_type(y_true, prot_attr)
    unpriv_group = get_unpriv_group(prot_attr, priv_group)

    true_positive_idx = np.where(y_true == pos_label)[0]  
    pred_positive_idx = np.where(y_pred == pos_label)[0]  
    unpriv_idx = np.where(prot_attr == unpriv_group)[0] if len(np.unique(prot_attr)) == 2 else np.where(prot_attr in unpriv_group)[0]
    pred_positive_unpriv_idx = np.intersect1d(pred_positive_idx, unpriv_idx)
    pred_positive_unpriv_true_pos_idx = np.intersect1d(pred_positive_unpriv_idx, true_positive_idx)
    PPV_unpriv_num = len(pred_positive_unpriv_true_pos_idx)
    PPV_unpriv_den = len(pred_positive_unpriv_idx)
    PPV_unpriv = PPV_unpriv_num / PPV_unpriv_den if  PPV_unpriv_den != 0 else PPV_unpriv_num / epsilon

    return PPV_unpriv

def positive_predicted_value_priv(y_true, y_pred, prot_attr, priv_group, pos_label):

    y_true, prot_attr = check_data_type(y_true, prot_attr)

    true_positive_idx = np.where(y_true == pos_label)[0]  
    pred_positive_idx = np.where(y_pred == pos_label)[0]  
    priv_idx = np.where(prot_attr == priv_group)[0] 
    pred_positive_priv_idx = np.intersect1d(pred_positive_idx, priv_idx)
    pred_positive_priv_true_pos_idx = np.intersect1d(pred_positive_priv_idx, true_positive_idx)
    PPV_priv_num = len(pred_positive_priv_true_pos_idx) 
    PPV_priv_den = len(pred_positive_priv_idx)
    PPV_priv = PPV_priv_num / PPV_priv_den if PPV_priv_den != 0 else PPV_priv_num / epsilon

    return PPV_priv

def positive_predicted_value_difference(y_true, y_pred, prot_attr, priv_group, pos_label):

    PPV_unpriv = positive_predicted_value_unpriv(y_true, y_pred, prot_attr, priv_group, pos_label)
    PPV_priv = positive_predicted_value_priv(y_true, y_pred, prot_attr, priv_group, pos_label)
    PPV_diff = PPV_unpriv - PPV_priv

    return PPV_diff

def positive_predicted_value_abs_difference(y_true, y_pred, prot_attr, priv_group, pos_label): # sufficiency
    
    PPV_diff = positive_predicted_value_difference(y_true, y_pred, prot_attr, priv_group, pos_label)
    PPV_abs_diff =  np.abs(PPV_diff)

    return PPV_abs_diff


def positive_predicted_value_ratio(y_true, y_pred, prot_attr, priv_group, pos_label):

    PPV_unpriv = positive_predicted_value_unpriv(y_true, y_pred, prot_attr, priv_group, pos_label)
    PPV_priv = positive_predicted_value_priv(y_true, y_pred, prot_attr, priv_group, pos_label)
    
    PPV_ratio = PPV_unpriv / PPV_priv if PPV_priv != 0 else PPV_unpriv / epsilon

    return PPV_ratio


#############################################################################################################################################


#############################################################################################################################################