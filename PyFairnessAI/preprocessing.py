
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from aif360.sklearn.preprocessing import Reweighing, ReweighingMeta
   
###############################################################################################################

class ReweighingMetaEstimator(BaseEstimator, ClassifierMixin):
   
    def __init__(self, estimator, prot_attr):
        
        self.estimator = estimator
        self.prot_attr = prot_attr

    def fit(self, X, y, sample_weight=None):

        # X must be a Pandas DataFrame
        
        A = X[self.prot_attr]
        reweigher_ = Reweighing(prot_attr=A)
        self.meta_estimator = ReweighingMeta(estimator=self.estimator, reweigher=reweigher_)
        
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
