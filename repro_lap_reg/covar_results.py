from fclsp.reshaping_utils import vec_hollow_sym

from repro_lap_reg.utils import merge_dicts
from repro_lap_reg.results_utils import compare_vecs, compare_adj_mats


def get_covar_results(est, true, zero_tol=0):
    """

    Parameters
    ----------
    est: an Estimator
        A covariance estimator.

    true: array-like, shape (n_features, n_features)

    zero_tol: float

    Output
    ------
    out: dict with keys 'utri' and 'graph'
    """

    covar_est = get_covar(est)

    est_utri = vec_hollow_sym(covar_est)
    true_utri = vec_hollow_sym(true)

    utri_results = compare_vecs(est=est_utri, truth=true_utri,
                                zero_tol=zero_tol)

    graph_results = compare_adj_mats(est=covar_est, truth=true,
                                     zero_tol=zero_tol)

    results = merge_dicts(utri_results, graph_results, allow_key_overlap=False)

    return results


def get_covar(estimator):
    if hasattr(estimator, 'covariance_'):
        return estimator.covariance_

    elif hasattr(estimator, 'best_estimator_'):
        return get_covar(estimator.best_estimator_)

    else:
        raise ValueError('No covariance matrix found')
