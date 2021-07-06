from fclsp.reshaping_utils import fill_hollow_sym
from repro_lap_reg.utils import merge_dicts
from repro_lap_reg.results_utils import compare_vecs, compare_adj_mats


def get_lin_reg_results(est, true, zero_tol=0):
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

    est_coef = get_coef(est)[0]

    est_adj = fill_hollow_sym(est_coef)
    true_adj = fill_hollow_sym(est_coef)

    coef_results = compare_vecs(est=est_coef, truth=true,
                                zero_tol=zero_tol)

    graph_results = compare_adj_mats(est=est_adj, truth=true_adj,
                                     zero_tol=zero_tol)

    results = merge_dicts(coef_results, graph_results, allow_key_overlap=False)

    return results


def get_coef(est):

    if hasattr(est, 'coef_'):
        return est.coef_, est.intercept_

    elif hasattr(est, 'best_estimator_'):
        return get_coef(est.best_estimator_)
