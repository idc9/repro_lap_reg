from numbers import Number
from sklearn.utils import check_random_state
import numpy as np


def sample_add_noise(means, scale=1.0, n_samples=100,
                     noise_dist='gauss', random_state=None):
    """
    Samples from an additive noise model, X_{ij} = mu_j + e_{ij},
    where m_j is the mean for the jth feature ane e_{ij} is a mean zero noise term.

    Parameters
    ----------
    means: array-like, shape (n_featuers, )
        The means.

    scale: float, array-like
        The noise scale. If a number is passed in, will use that value for each feature. Can pass in an array to privde features with their own noise values.

        If 'noise_dist' == 'gauss' then scale is the standard deviation.

    n_samples: int
        Number of samples to draw.

    noise_dist: str
        Which distribution to sample the noise from. Must be one of ['guass']

    random_state: None, int
        Seed.

    Output
    ------
    X: array-like, shape (n_samples, n_features)
        The data samples.
    """

    assert noise_dist in ['gauss'] # TODO: add more
    rng = check_random_state(random_state)

    means = np.array(means).reshape(-1)
    n_features = len(means)

    # setup scales
    if isinstance(scale, Number):
        scale = np.ones(len(means)) * scale
    assert len(scale) == len(means)

    if noise_dist == 'gauss':
        E = rng.normal(size=(n_samples, n_features))
        E = E @ np.diag(scale)
        X = means + E

    return X
