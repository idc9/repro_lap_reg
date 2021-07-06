from sklearn.covariance import EmpiricalCovariance
from sklearn.base import BaseEstimator
import numpy as np

from ya_glm.autoassign import autoassign


class NearestCovarFixedSupport(BaseEstimator):

    @autoassign
    def __init__(self, assume_centered=False, store_precision=True,
                 score_metric='L2'):
        pass

    def fit(self, X, y=None, support=None):
        """

        Parameters
        ----------
        X: array-like, (n_samples, n_features)

        y: None

        support: None, array-like shape (n_features, n_features)
        """
        assert y is None  # so we dont accidently pass in a y

        n_features = X.shape[1]

        # fit empirical covariance estimator
        est = EmpiricalCovariance(assume_centered=self.assume_centered,
                                  store_precision=self.store_precision)
        est.fit(X)
        covar = est.covariance_

        # set
        if support is not None:
            support = np.array(support).astype(bool)
            assert support.shape == (n_features, n_features)

            # all diagonal elements should be non-zero
            assert all(np.diag(support))

            # support should be symmetric
            assert all((support.T == support).ravel())

            # zero all elements not in support
            covar[~support] = 0.0

        # self._set_estimate(covariance=covar, location=est.location_)
        self.covariance_ = covar
        self.location_ = est.location_

        return self
