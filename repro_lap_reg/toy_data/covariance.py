import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import diags
from itertools import combinations
from sklearn.utils import check_random_state
import inspect

from repro_lap_reg.toy_data.adj_mats import complete_graph
from fclsp.linalg_utils import eigh_wrapper


def get_bd_covar_from_args(block_sampler='dense', block_sizes=[5, 5, 5],
                           kws={}):
    """
    Gets a block diagonal covariace matrix.

    Parameters
    ----------
    block_sampler: str
        How the block should be sampled. Must be one of ['dense', 'ar1']. See repro_fclsp.toy_data.covariance() for details.

    block_sizes: list of ints
        Sizes of each block.

    kws: dict
        Key word arguments for the block sampling functions.
    """

    assert block_sampler in ['dense', 'ar1']

    if block_sampler == 'dense':
        cov_getter = cov_dense

    elif block_sampler == 'ar1':
        cov_getter = cov_ar1

    return block_diag_cov(cov_getter=cov_getter,
                          block_sizes=block_sizes,
                          **kws)


def block_diag_cov(cov_getter, block_sizes=[5, 5, 5], random_state=None,
                   **kws):
    """
    Parameters
    ----------
    cov_getter: callable(n_features, *args, **kwargs)
        A function that takes the number of features and returns a covariance matrix.

    block_sizes:
        Number of features in each block.

    random_state: int, None
        Seed.

    *args, **kwargs: arguments to cov_getter

    Output
    ------
    cov: array-like, (sum(n_features), sum(n_features))
        The block diagonal covariance matrix.
    """
    has_rand_state_arg = 'random_state' in inspect.getargspec(diags).args

    if has_rand_state_arg:
        rng = check_random_state(random_state)
        kws['random_state'] = rng

    covs = [cov_getter(n_features=block_sizes[b], **kws)
            for b in range(len(block_sizes))]
    return block_diag(*covs)


def cov_dense(n_features=100, scale=0.5,
              edges='ones', pos=True, force_psd=True, random_state=None):
    """
    Returns a covariance matrix with a constant diagonal and whose off diagnale elements are obtained from adj_mats.complete_graph()

    Parameters
    ----------
    n_features: int

    scale: float
        Scale of the off diagonal entries.

    edges: str
        How the edges should be sampled. See adj_mats.complete_graph()

    pos: bool
        Should the off-diagonal entries be all positive.

    force_psd: bool
        Make sure the covariance matrix is positive semi-definite zeroing out all negative eigenvalues.

    random_state: None, int
        Random seed for sampling.

    Output
    ------
    cov: array-like, (n_features, n_features)
        The sampled covariance matrix.
    """
    cov = complete_graph(n_nodes=n_features, edges=edges,
                         pos=pos, random_state=random_state)

    cov = cov * scale

    np.fill_diagonal(cov, 1.0)

    if force_psd:
        cov = project_psd(cov)

    return cov


def project_psd(X):
    """
    Kills the negative eigenvalues.
    """
    evals, evecs = eigh_wrapper(X)
    evals = np.clip(evals, a_min=0, a_max=np.inf)
    return evecs @ diags(evals) @ evecs.T


def cov_ar1(n_features=20, rho=0.5):
    """
    Returns a covariance matrix whose entries are Sigma_{ij} = rho^{|i - j|}.

    Parameters
    ----------
    n_features: int
        Dimension of the random vector.

    rho: float

    Output
    ------
    cov: array-like, (n_features, n_features)
        The covariance matrix.
    """
    cov = np.ones((n_features, n_features))
    for (i, j) in combinations(range(n_features), 2):
        cov[i, j] = cov[j, i] = rho ** abs(i - j)

    return cov
