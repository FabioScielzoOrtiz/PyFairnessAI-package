# PyFairnessAI

## Introduction

A package for applying a Fairness Machine Learning Workflow (*), focused in classification problems.

The framework includes diverse fairness metrics and processors, as well as model selection algorithms based on both fairness and predictive power. Additionally, this framework allows to combine multiple fairness processors as well as multiple sensitive variables (protected attributes).

This package is mainly inspired in `aif360` and has been designed to work with `scikit-learn`.

(*) Machine Learning Workflow refers to the process of finding the best model for a certain Machine Learning problem and data.

## Modules

`PyFairnessAI` has diverse functionalities grouped in the following modules::

- `metrics`: a set of fairness metrics which can be used to asses the fairness degree of classification algorithms. Most of them are based on `aif360` developments.

   - `statistical_parity_difference`
   - `abs_statistical_parity_difference`
   - `equal_opportunity_difference`
   - `abs_equal_opportunity_difference`
   - `disparate_impact_ratio`
   - `average_odds_error`
   - `false_positive_rate_difference`
   - `false_negative_rate_difference`
   - `true_positive_rate_difference`
   - `true_negative_rate_difference`
   - `false_positive_rate_ratio`
   - `false_negative_rate_ratio`
   - `true_positive_rate_ratio`
   - `true_negative_rate_ratio`
   - `positive_predicted_value_difference`
   - `positive_predicted_value_abs_difference`
   - `positive_predicted_value_ratio`

  All the metrics are functions with the following arguments: 

  - `y_true`: array with the true values of the response (target) variable.

  - `y_pred`: array with the predicted values of the response (target) variable.

  - `prot_attr`: array with the values of the sensitive variable (protected attribute).

  - `priv_group`: label of the privileged group in the sensitive variable (protected attribute).

  - `pos_label`: label of the positive (favorable) group in the response variable.

  
- `preprocessing`: a set of fairness pre-processors.

  - `ReweighingMetaEstimator`: a `scikit-learn` like meta estimator, which applies reweighing fairness pre-processing to the input data before fitting the provided `scikit-learn` estimator.



- `inprocessing`: a set of fairness in-processors.

  - `AdversarialDebiasingEstimator`: a `scikit-learn` like estimator, designed with fairness principles under the hood.

  - `ExponentiatedGradientReductionMetaEstimator`: a `scikit-learn` like meta estimator, which applies exponential gradient reduction as fairness in-processing over a provided `scikit-learn` estimator.

  - `GridSearchReductionMetaEstimator`: a  `scikit-learn` like meta estimator, which applies grid search reduction as fairness in-processing over a provided `scikit-learn` estimator.

- `postprocessing`: a set of fairness pre-processors.

   - `PostProcessingMetaEstimator`: a  `scikit-learn` like meta estimator, which applies `CalibratedEqualizedOdds` or `RejectOptionClassifier` as fairness post-processing over the output of a provided `scikit-learn` estimator.


- `model_selection`: a set of model selection algorithms based on both fairness and predictive power principles.

   - `cross_val_score_fairness`: a `scikit-learn` like cross-validation score based on fairness metrics.

   - `combined_score`: an function that combines fairness and predictive scores by mean of a weighted normalized average.

   - `RandomizedSearchCVFairness`: a `scikit-learn` like randomized search cross-validation which allows to apply hyper-parameter optimization based on a fairness score, a predictive score, or the combined score. 

## Advantages

The main advantages of `PyFairnessAI` with respect to other similar frameworks, like `aif360`, are the following:

- A combined metric approach to measure both the fairness and predictive power of a classification model at the same time.

- Hyper-parameter optimization of classification algorithms based on both fairness degree and predictive power.

- Can be used for finding the best classification model based on both fairness and predictive scores. 

- Allows multi fairness processors and multi sensitive variables (protected attributes).

- Designed to work with `scikit-learn` objects.

## Demo

To do

 

