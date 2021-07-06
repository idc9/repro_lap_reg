from sklearn.covariance import EmpiricalCovariance
from sklearn.utils import check_random_state

import numpy as np

from fclsp.thresh.Covariance import ThreshCV
from fclsp.bd_thresh.Covariance import FclspLLACV

from fclsp.reshaping_utils import vec_hollow_sym

from repro_lap_reg.constrained.NearestCovarFixedSupport import \
    NearestCovarFixedSupport
from repro_lap_reg.covar_results import get_covar_results
# from repro_lap_reg.toy_data.samplers import get_mv_gaussian_sampler
from repro_lap_reg.toy_data.covariance import get_bd_covar_from_args
from repro_lap_reg.utils import merge_dicts
from repro_lap_reg.sim import run_sim
from repro_lap_reg.sim_from_args import get_block_sizes_from_config, \
    get_pen_kws_from_config
from repro_lap_reg.means_est_sim import get_tune_path_info


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

    # true covariance
    true_cov_kws = {'block_sampler': 'dense',
                    'kws': {'scale': args.support_scale},
                    'block_sizes': block_sizes,  # [2] * 5,  # [5, 5, 5]
                    }

    # generalized thresholding
    gen_thresh_kws = {'n_pen_vals': args.n_pen_vals,
                      'pen_min_mult': args.pen_min_mult_fcp,
                      'cv': args.cv,
                      'cv_select_rule': args.cv_select_rule,
                      'cv_n_jobs': args.cv_n_jobs,
                      }

    # penalty function + cross-validation
    fclsp_kws, _ = get_pen_kws_from_config(args)

    ####################################
    # setup parameters and sample data #
    ####################################

    rng = check_random_state(args.data_seed)

    # setup true covariance matrix and sample data
    true_cov = get_bd_covar_from_args(**true_cov_kws)
    # data_sampler = get_mv_gaussian_sampler(cov=true_cov)
    # X = data_sampler(n_samples=args.n_samples, random_state=args.data_seed)
    X = rng.multivariate_normal(mean=np.zeros(true_cov.shape[0]),
                                cov=true_cov,
                                size=args.n_samples)

    ###############
    # initializer #
    ###############

    default_init_est = ThreshCV(thresh='hard', **gen_thresh_kws)
    default_init_est.set_params(cv=10)  # always make this 10
    default_init_est.fit(X)

    empir_init_est = EmpiricalCovariance()
    empir_init_est.fit(X)

    ################
    # setup models #
    ################

    zero_init = {'values': np.zeros_like(vec_hollow_sym(true_cov))}

    lla_strategies = [('init=0__steps=3',
                       {'init': 'zero', 'lla_n_steps': 3}),

                      ('init=0__steps=convergence',
                       {'init': 'zero',
                        'lla_n_steps': args.n_lla_steps_max}),

                      ('init=default__steps=2',
                       {'init': 'default', 'lla_n_steps': 2}),

                      ('init=default__steps=convergence',
                       {'init': 'default',
                        'lla_n_steps': args.n_lla_steps_max}),

                      ('init=empirical__steps=2',
                       {'init': 'empirical', 'lla_n_steps': 2}),

                      ('init=empirical__steps=convergence',
                       {'init': 'empirical',
                        'lla_n_steps': args.n_lla_steps_max})
                      ]

    models = {}
    models['empirical'] = EmpiricalCovariance()

    # Genralized thresholding models
    for kind in ['hard', 'soft']:  # 'adpt_lasso',
        models['thresh__kind={}'.format(kind)] = \
            ThreshCV(thresh=kind, **gen_thresh_kws)

    # FLCP models
    for (stub, strat_kws) in lla_strategies:
        # set initializations
        if strat_kws['init'] == 'default':
            strat_kws['init'] = default_init_est

        elif strat_kws['init'] == 'empirical':
            strat_kws['init'] = empir_init_est

        elif strat_kws['init'] == 'zero':
            strat_kws['init'] = zero_init

        models['fclsp__{}'.format(stub)] = \
            FclspLLACV(**merge_dicts(fclsp_kws, strat_kws))

    out, models = single_sim_from_data(X=X, models=models, true_cov=true_cov,
                                       verbosity=verbosity,
                                       zero_tol=args.results_zero_tol)

    out['args'] = args

    return out, models


def single_sim_from_data(X, models, true_cov, verbosity=1,
                         zero_tol=1e-8):
    """
    Runs a single simulation on a dataset.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The dataset.

    models: dict of estimators.
        The competing models we want to fit.

    true_cov: array-like, (n_features, n_features)
        The true covariance matrix.

    verbosity: int
        How much output to printout.

    zero_tol: float
        Zero tolerance for determining if a value is zero or not.
        Some optimization solvers may not return exact zeros.

    Output
    ------
    out: dict
        All the output data.

    models: dict
        The fitted models.
    """
    return run_sim(get_est_results=get_covar_results,
                   get_tune_path_info=get_tune_path_info,
                   get_oracle_est=get_covar_oracle_est,
                   models=models,
                   true_param=true_cov,
                   X=X,
                   y=None,
                   verbosity=verbosity,
                   zero_tol=zero_tol)


def get_covar_oracle_est(true_support, X, y=None):
    """
    Gets the fitted oracle estimator and block diagonal oracle estimate.

    Parameters
    ----------
    true_support: array-like, shape (n_features, n_features)
        The mask of non-zero entries of the true covariance matrix.

    X: array-like, shape (n_samples, n_features)
        The training data.

    Output
    ------
    oracle_cov, oracle_model

    oracle_cov: array-like, shape (n_features, )
        The oracle covariance estimate.

    oracle_model: estimator
        The fitted oracle estimator.
    """

    # Oracle
    oracle_model = NearestCovarFixedSupport()
    oracle_model.fit(X=X, support=true_support)

    oracle_cov = oracle_model.covariance_

    return oracle_cov, oracle_model
