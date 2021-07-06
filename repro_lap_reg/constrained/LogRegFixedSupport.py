import numpy as np
from ya_glm.backends.fista.LogisticRegression import Vanilla

from repro_lap_reg.constrained.utils import put_back


class LogRegFixedSupport(Vanilla):

    def fit(self, X, y, support=None):
        """

        Parameters
        ----------
        X: array-like, (n_samples, n_features)

        X: array-like, (n_samples, )

        support: None, array-like shape (n_features, n_features)
        """

        if support is not None:
            support = np.array(support).astype(bool).reshape(-1)
            assert support.shape == (X.shape[1], )

            super().fit(X=X[:, support], y=y)

            self.coef_ = put_back(values_S=self.coef_, support=support)

        else:
            super().fit(X=X, y=y)

        return self
