###############################################################################################################

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from aif360.sklearn.inprocessing import (AdversarialDebiasing, ExponentiatedGradientReduction, GridSearchReduction)
from fairlearn.reductions import Moment # necessary for ExponentiatedGradientReduction

###############################################################################################################

class AdversarialDebiasingEstimator(BaseEstimator, ClassifierMixin):
    
    def __init__(self, prot_attr, scope_name='classifier', adversary_loss_weight=0.1, num_epochs=50, 
                 batch_size=128, classifier_num_hidden_units=200, debias=True, verbose=False, 
                 random_state=None):
        
        self.prot_attr = prot_attr
        self.scope_name = scope_name
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        
        # Extract protected attribute
        A = X[self.prot_attr]

        # Initialize the original AdversarialDebiasing model
        self.estimator = AdversarialDebiasing(prot_attr=A,
                                              scope_name=self.scope_name,
                                              adversary_loss_weight=self.adversary_loss_weight,
                                              num_epochs=self.num_epochs,
                                              batch_size=self.batch_size,
                                              classifier_num_hidden_units=self.classifier_num_hidden_units,
                                              debias=self.debias,
                                              verbose=self.verbose,
                                              random_state=self.random_state)
        
        # Handle sample_weight if provided
        if sample_weight is not None:
            # Normalize the weights if necessary (ensure they sum to 1 or another form)
            sample_weight = sample_weight / sample_weight.sum()
            # Resample X and y based on sample_weight
            X_resampled, y_resampled = self._resample_with_weights(X, y, sample_weight)
            self.estimator.fit(X_resampled, y_resampled)
        else:
            self.estimator.fit(X, y)
        
        self.classes_ = self.estimator.classes_

        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def _resample_with_weights(self, X, y, sample_weight):
        """Resample dataset according to sample weights."""
        import numpy as np

        # Scale sample weights to integer values for resampling
        weights = np.round(sample_weight * len(sample_weight)).astype(int)

        # Resample X and y
        if isinstance(X, np.ndarray):
            X_resampled = np.repeat(X, weights, axis=0)
        elif isinstance(X, pd.DataFrame):
            X_resampled = pd.DataFrame(np.repeat(X.values, weights, axis=0), columns=X.columns)
        if isinstance(y, np.ndarray):
            y_resampled = np.repeat(y, weights, axis=0)
        elif isinstance(y, pd.Series):
            y_resampled = pd.Series(np.repeat(y.values, weights), name=y.name)

        return X_resampled, y_resampled

###############################################################################################################

class ExponentiatedGradientReductionMetaEstimator(BaseEstimator, ClassifierMixin):
    
    def __init__(self, prot_attr, estimator, constraints='ErrorRateParity', eps=0.01, max_iter=50, 
                 nu=None, eta0=2.0, run_linprog_step=True, drop_prot_attr=True):

        self.prot_attr = prot_attr
        self.estimator = estimator
        self.constraints = constraints
        self.eps = eps
        self.max_iter = max_iter
        self.nu = nu
        self.eta0 = eta0
        self.run_linprog_step = run_linprog_step
        self.drop_prot_attr = drop_prot_attr

    def fit(self, X, y, sample_weight=None):
        
        # Initialize the original AdversarialDebiasing model
        self.meta_estimator = ExponentiatedGradientReduction(prot_attr=self.prot_attr, estimator=self.estimator, 
                                                             constraints=self.constraints, eps=self.eps, max_iter=self.max_iter, 
                                                             nu=self.nu, eta0=self.eta0, run_linprog_step=self.run_linprog_step, 
                                                             drop_prot_attr=self.drop_prot_attr)
        
        # Handle sample_weight if provided
        if sample_weight is not None:
            # Normalize the weights if necessary (ensure they sum to 1 or another form)
            sample_weight = sample_weight / sample_weight.sum()
            # Resample X and y based on sample_weight
            X_resampled, y_resampled = self._resample_with_weights(X, y, sample_weight)
            self.meta_estimator.fit(X_resampled, y_resampled)
        else:
            self.meta_estimator.fit(X, y)
        
        self.classes_ = self.meta_estimator.classes_

        return self

    def predict(self, X):
        return self.meta_estimator.predict(X)

    def predict_proba(self, X):
        return self.meta_estimator.predict_proba(X)

    def _resample_with_weights(self, X, y, sample_weight):
        """Resample dataset according to sample weights."""
        import numpy as np

        # Scale sample weights to integer values for resampling
        weights = np.round(sample_weight * len(sample_weight)).astype(int)

        # Resample X and y
        if isinstance(X, np.ndarray):
            X_resampled = np.repeat(X, weights, axis=0)
        elif isinstance(X, pd.DataFrame):
            X_resampled = pd.DataFrame(np.repeat(X.values, weights, axis=0), columns=X.columns)
        if isinstance(y, np.ndarray):
            y_resampled = np.repeat(y, weights, axis=0)
        elif isinstance(y, pd.Series):
            y_resampled = pd.Series(np.repeat(y.values, weights), name=y.name)

        return X_resampled, y_resampled

###############################################################################################################

class GridSearchReductionMetaEstimator(BaseEstimator, ClassifierMixin):
    
    def __init__(self, prot_attr, estimator, constraints='ErrorRateParity', constraint_weight=0.5, 
                       grid_size=10, grid_limit=2.0, grid=None, drop_prot_attr=True, loss='ZeroOne', 
                       min_val=None, max_val=None):
        
        self.prot_attr = prot_attr
        self.estimator = estimator
        self.constraints = constraints
        self.constraint_weight = constraint_weight
        self.grid_size = grid_size
        self.grid_limit = grid_limit
        self.grid = grid
        self.loss = loss
        self.min_val = min_val
        self.max_val = max_val
        self.drop_prot_attr = drop_prot_attr

    def fit(self, X, y, sample_weight=None):
        
        # Initialize the original AdversarialDebiasing model
        self.meta_estimator = GridSearchReduction(prot_attr=self.prot_attr, estimator=self.estimator, 
                                                  constraints=self.constraints, constraint_weight=self.constraint_weight, 
                                                  grid_size=self.grid_size, grid_limit=self.grid_limit, grid=self.grid, 
                                                  drop_prot_attr=self.drop_prot_attr, loss=self.loss, 
                                                  min_val=self.min_val, max_val=self.max_val)
        
        # Handle sample_weight if provided
        if sample_weight is not None:
            # Normalize the weights if necessary (ensure they sum to 1 or another form)
            sample_weight = sample_weight / sample_weight.sum()
            # Resample X and y based on sample_weight
            X_resampled, y_resampled = self._resample_with_weights(X, y, sample_weight)
            self.meta_estimator.fit(X_resampled, y_resampled)
        else:
            self.meta_estimator.fit(X, y)
        
        self.classes_ = self.meta_estimator.classes_

        return self

    def predict(self, X):
        return self.meta_estimator.predict(X)

    def predict_proba(self, X):
        return self.meta_estimator.predict_proba(X)

    def _resample_with_weights(self, X, y, sample_weight):
        """Resample dataset according to sample weights."""
        import numpy as np

        # Scale sample weights to integer values for resampling
        weights = np.round(sample_weight * len(sample_weight)).astype(int)

        # Resample X and y
        if isinstance(X, np.ndarray):
            X_resampled = np.repeat(X, weights, axis=0)
        elif isinstance(X, pd.DataFrame):
            X_resampled = pd.DataFrame(np.repeat(X.values, weights, axis=0), columns=X.columns)
        if isinstance(y, np.ndarray):
            y_resampled = np.repeat(y, weights, axis=0)
        elif isinstance(y, pd.Series):
            y_resampled = pd.Series(np.repeat(y.values, weights), name=y.name)

        return X_resampled, y_resampled

###############################################################################################################
