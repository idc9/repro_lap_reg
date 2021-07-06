from fclsp.reshaping_utils import vec_hollow_sym, fill_hollow_sym

from repro_lap_reg.utils import merge_dicts
from repro_lap_reg.results_utils import compare_vecs, compare_adj_mats


def get_means_est_results(est, true, zero_tol=0):
    """

    Parameters
    ----------
    est: an Estimator
        A means estimator.

    true: array-like, shape (n_features, n_features)

    zero_tol: float

    Output
    ------
    out: dict with keys 'utri' and 'graph'
    """

    means_est_vec = get_means(est)
    means_est_mat = fill_hollow_sym(means_est_vec)

    true_utri = vec_hollow_sym(true)

    vec_results = compare_vecs(est=means_est_vec, truth=true_utri,
                               zero_tol=zero_tol)

    graph_results = compare_adj_mats(est=means_est_mat, truth=true,
                                     zero_tol=zero_tol)

    results = merge_dicts(vec_results, graph_results, allow_key_overlap=False)

    return results


def get_means(est):
    if hasattr(est, 'means_'):
        return est.means_

    elif hasattr(est, 'best_estimator_'):
        return get_means(est.best_estimator_)

    else:
        raise ValueError("Unable to find the mean")
