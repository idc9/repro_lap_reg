from sklearn.utils import check_random_state
import numpy as np

from ya_glm.backends.fista.LogisticRegression import Ridge, \
    LassoCV, FcpLLACV  # Vanilla

from fclsp.glm.fista.LogisticRegression import FclspLLACV
from fclsp.reshaping_utils import vec_hollow_sym

from repro_lap_reg.toy_data.regression import sample_X, sample_log_reg_response
from repro_lap_reg.toy_data.adj_mats import block_diag_complete
from repro_lap_reg.constrained.LogRegFixedSupport import \
    LogRegFixedSupport
from repro_lap_reg.lin_reg_results import get_lin_reg_results
from repro_lap_reg.utils import merge_dicts
from repro_lap_reg.sim import run_sim
from repro_lap_reg.sim_from_args import get_block_sizes_from_config, \
    get_pen_kws_from_config
from repro_lap_reg.means_est_sim import get_tune_path_info

# we add a small ridge penalty to the logistic regression models because
# the unpenalized models can be unstable
DEFAULT_RIDGE = .01


def single_sim_from_args(args, verbosity=1):
    """
    Runs a single simulation from an args object: sample a dataset, fit competing models, the compute performance metrics.

    Output
    ------
    out: dict
        All the output data.

    models: dict
        The fitted models.
    """

    ##########################
    # Setup config from args #
    ##########################
    block_sizes = get_block_sizes_from_config(args)

    # penalty function + cross-validation
    fclsp_kws, fcp_kws = get_pen_kws_from_config(args)

    # possibly add ridge penalty
    if args.add_ridge:
        fclsp_kws['opt_kws'] = {'L2_pen': DEFAULT_RIDGE}
        fcp_kws['opt_kws'] = {'L2_pen': DEFAULT_RIDGE}

    ####################################
    # setup parameters and sample data #
    ####################################

    true_coef = block_diag_complete(block_sizes=block_sizes,
                                    edges='ones',
                                    pos=False,
                                    random_state=234)

    true_coef = vec_hollow_sym(true_coef)
    true_intercept = 0

    rng = check_random_state(args.data_seed)
    X = sample_X(n_samples=args.n_samples, n_features=len(true_coef),
                 X_dist=args.X_dist, x_corr=args.X_corr_strength,
                 random_state=rng)

    y, p = sample_log_reg_response(X=X, coef=true_coef,
                                   intercept=true_intercept,
                                   random_state=rng)

    ###############
    # initializer #
    ###############
    lasso_kws = {k: v for (k, v) in fcp_kws.items()
                 if k not in ['pen_func', 'pen_func_kws'] and 'lla' not in k}

    default_init_est = LassoCV(**lasso_kws)
    # default_init_est.set_params(cv=10)
    default_init_est.cv = 10
    default_init_est.fit(X, y)

    ################
    # setup models #
    ################

    zero_init = {'coef': np.zeros_like(true_coef), 'intercept': 0}

    # lla_strategies_fclsp = [('init=0__steps=3',
    #                         {'init': 'zero', 'lla_n_steps': 3}),

    #                         ('init=0__steps=convergence',
    #                          {'init': 'zero',
    #                           'lla_n_steps': args.n_lla_steps_max}),

    #                         ('init=default__steps=2',
    #                          {'init': 'default', 'lla_n_steps': 2}),

    #                         ('init=default__steps=convergence',
    #                         {'init': 'default',
    #                          'lla_n_steps': args.n_lla_steps_max})
    #                         ]

    # lla_strategies_fcp = [('init=0__steps=2',
    #                        {'init': 'zero', 'lla_n_steps': 2}),

    #                       ('init=0__steps=convergence',
    #                        {'init': 'zero',
    #                         'lla_n_steps': args.n_lla_steps_max}),

    #                       ('init=default__steps=1',
    #                        {'init': 'default', 'lla_n_steps': 1}),

    #                       ('init=default__steps=convergence',
    #                        {'init': 'default',
    #                         'lla_n_steps': args.n_lla_steps_max})

    #                       ]

    lla_strategies_fclsp = [
                            ('init=default__steps=2',
                             {'init': 'default', 'lla_n_steps': 2}),
                            ]

    lla_strategies_fcp = [
                          ('init=default__steps=1',
                           {'init': 'default', 'lla_n_steps': 1}),
                          ]
    models = {}

    # if X.shape[0] < .95 * X.shape[1]:
    #     # if we are in high dimensions then let's just use a light ridge penalty
    models['vanilla'] = Ridge(pen_val=DEFAULT_RIDGE)
    # else:
    #     models['vanilla'] = Vanilla()

    models['lasso'] = LassoCV(**lasso_kws)

    # FLCP models
    for (stub, strat_kws) in lla_strategies_fclsp:

        # set initialization
        if strat_kws['init'] == 'default':
            strat_kws['init'] = default_init_est

        elif strat_kws['init'] == 'zero':
            strat_kws['init'] = zero_init

        models['fclsp__{}'.format(stub)] = \
            FclspLLACV(**merge_dicts(fclsp_kws, strat_kws))

    # FCP models
    for (stub, strat_kws) in lla_strategies_fcp:

        # set initialization
        if strat_kws['init'] == 'default':
            strat_kws['init'] = default_init_est

        elif strat_kws['init'] == 'zero':
            strat_kws['init'] = zero_init

        models['fcp__{}'.format(stub)] = \
            FcpLLACV(**merge_dicts(fcp_kws, strat_kws))

    out, models = single_sim_from_data(X=X, y=y, models=models,
                                       true_coef=true_coef,
                                       verbosity=verbosity,
                                       zero_tol=args.results_zero_tol,
                                       add_ridge=args.add_ridge)

    out['args'] = args

    return out, models


def single_sim_from_data(X, y, models,
                         true_coef,  # true_intercept,
                         verbosity=1,
                         zero_tol=1e-8,
                         add_ridge=False):
    """
    Runs a single simulation on a dataset.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The covariate data.

    y: array-like, shape (n_samples,)
        The response.

    models: dict of estimators.
        The competing models we want to fit.

    true_coef: array-like, (n_features, )
        The true regression coefficient.

    verbosity: int
        How much output to printout.

    zero_tol: float
        Zero tolerance for determining if a value is zero or not.
        Some optimization solvers may not return exact zeros.


    add_ridge: bool
        Whether or not to add a ridge penalty to each logistic regression model.

    Output
    ------
    out: dict
        All the output data.

    models: dict
        The fitted models.
    """

    if add_ridge:
        _get_oracle_est = get_oracle_est_with_ridge

    else:
        _get_oracle_est = get_oracle_est

    return run_sim(get_est_results=get_lin_reg_results,
                   get_tune_path_info=get_tune_path_info,
                   get_oracle_est=_get_oracle_est,
                   models=models,
                   true_param=true_coef,
                   X=X,
                   y=y,
                   verbosity=verbosity,
                   zero_tol=zero_tol)


def get_oracle_est(true_support, X, y):
    """
    Gets the fitted oracle estimator and block diagonal oracle estimate.

    Parameters
    ----------
    true_support: array-like, shape (n_features, )
        The mask of non-zero entries of the true coefficient.

    X: array-like, shape (n_samples, n_features)
        The training data.

    Output
    ------
    oracle_coef, oracle_model

    oracle_coef: array-like, shape (n_features, )
        The oracle means estimate.

    oracle_model: estimator
        The fitted oracle estimator.
    """

    # Oracle
    oracle_model = LogRegFixedSupport()
    oracle_model.fit(X=X, y=y, support=true_support)

    oracle_coef = oracle_model.coef_
    return oracle_coef, oracle_model


def get_oracle_est_with_ridge(true_support, X, y):
    """
    Gets the fitted oracle estimator and block diagonal oracle estimate. Here the oracle model includes a ridge penalty.

    Parameters
    ----------
    true_support: array-like, shape (n_features, )
        The mask of non-zero entries of the true coefficient.

    X: array-like, shape (n_samples, n_features)
        The training data.

    Output
    ------
    oracle_coef, oracle_model

    oracle_coef: array-like, shape (n_features, )
        The oracle means estimate.

    oracle_model: estimator
        The fitted oracle estimator.
    """

    # Oracle
    oracle_model = LogRegFixedSupport(opt_kws={'L2_pen': DEFAULT_RIDGE})
    oracle_model.fit(X=X, y=y, support=true_support)

    oracle_coef = oracle_model.coef_
    return oracle_coef, oracle_model
