###############################################################################################################

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset

######################################################################################################################################################################

def privileged_groups_sens(data_aif):
    return dict(zip(data_aif.protected_attribute_names, [round(x[0]) for x in data_aif.privileged_protected_attributes])) 

def unprivileged_groups_sens(data_aif):
    return dict(zip(data_aif.protected_attribute_names, [round(x[0]) for x in data_aif.unprivileged_protected_attributes])) 

######################################################################################################################################################################

def inv_logit(x):
    return np.exp(x)/(1+np.exp(x))

def binary_data_simulation(n, p_sens, p_no_sens, mean, cov, gamma, beta, random_state, output_type='aif360'): 

    np.random.seed(seed=random_state)

    Z = np.random.multivariate_normal(mean, cov, size=n) # Z = (A,X)
    Z_prob = inv_logit(Z)
    Z = np.random.binomial(1, Z_prob)

    coef = np.concatenate((gamma, beta)) # vector with coef for generating Y. Firsts ones (gamma) are for sens variables, remaining (beta) for non sens.
    Y = Z @ coef.T # Z = (A,X). It's assumed that first p_sens columns of Z are the sens predictors (A), and remaining p_no_sens are the non sens ones (X).
    # np.array_equal(X @ coef.T, coef.dot(X.T), coef @ X.T) --> True
    Y_prob = inv_logit(Y)
    Y = np.random.binomial(1, Y_prob)

    data_arr = np.concatenate((Y.reshape((n,1)), Z), axis=1) # concatenating by columns
    data_df = pd.DataFrame(data_arr)
    response_name = ['Y']
    sens_names = [f'A_{j+1}' for j in range(p_sens)]
    no_sens_names = [f'X_{j+1}' for j in range(p_no_sens)]
    data_col_names = response_name + sens_names + no_sens_names # Y: response. A: sens predictors. X: non sens predictors.
    data_df.columns = data_col_names

    if output_type == 'aif360':

        data_aif = BinaryLabelDataset(df=data_df, 
                                      label_names=np.array(response_name), # Response name
                                      favorable_label=1, # Response favorable label
                                      unfavorable_label=0, # Response unfavorable label
                                      protected_attribute_names = np.array(sens_names) # Sens variable/s name/s
                                      )
        
        # Adding new properties to the BinaryLabelDataset class
        data_aif.privileged_groups_sens = privileged_groups_sens(data_aif)
        data_aif.unprivileged_groups_sens = unprivileged_groups_sens(data_aif)

        return data_aif

    elif output_type == 'pandas':

        return data_df


######################################################################################################################################################################


######################################################################################################################################################################