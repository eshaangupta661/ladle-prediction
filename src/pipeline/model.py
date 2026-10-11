"""Estimators that predict ladle life as the campaign heat counter plus a learned correction.

Life grows without bound within a campaign, so tree models trained on raw life cannot
predict beyond the largest life seen in training. Learning only the correction to the
counter keeps predictions valid for long campaigns.
"""
from sklearn.base import BaseEstimator, RegressorMixin, clone

BASELINE_COLUMN = "expected_life"


class CounterBaseline(BaseEstimator, RegressorMixin):
    """Reference model: life equals starting life plus heats observed in the campaign."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[BASELINE_COLUMN].to_numpy(dtype=float)


class CounterResidualRegressor(BaseEstimator, RegressorMixin):
    """Fit ``estimator`` on (life - counter) and add the counter back at prediction time."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y - X[BASELINE_COLUMN].to_numpy(dtype=float))
        return self

    def predict(self, X):
        return X[BASELINE_COLUMN].to_numpy(dtype=float) + self.estimator_.predict(X)
