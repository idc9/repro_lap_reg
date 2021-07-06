import numpy as np
from sklearn.utils import check_random_state
from scipy.special import expit


def sample_lin_reg_response(X, coef, intercept=0,
                            noise_std=1.0, random_state=None):
    rng = check_random_state(random_state)
    y = X @ coef + intercept + noise_std * rng.normal(size=X.shape[0])
    return y


def sample_log_reg_response(X, coef, intercept=0, random_state=None):
    rng = check_random_state(random_state)
    z = X @ coef + intercept
    p = expit(z)
    y = rng.binomial(n=1, p=p)

    return y, p


def sample_X(n_samples=100, n_features=10, X_dist='indep', x_corr=0.1,
             random_state=None):
    """
    Samples a random design matrix.

    Parameters
    ----------
    n_samples: int
        Number of samples to draw.

    n_features: int
        Number of features.

    X_dist: str
        How to sample the X data. Must be one of ['indep', 'corr'].
        X data is always follows a multivariate Gaussian.
        If 'corr', then cov = (1 - corr) * I + corr * 11^T.

    x_corr: float
        How correlated the x data are.

    random_state: None, int
        The seed.

    Output
    ------
    X: array-like, (n_samples, n_features)

    """
    rng = check_random_state(random_state)

    assert X_dist in ['indep', 'corr']

    # sample X data
    if X_dist == 'indep':
        return rng.normal(size=(n_samples, n_features))

    elif X_dist == 'corr':
        cov = (1 - x_corr) * np.eye(n_features) + \
            x_corr * np.ones((n_features, n_features))

        return rng.multivariate_normal(mean=np.zeros(n_features),
                                       size=n_samples, cov=cov)
