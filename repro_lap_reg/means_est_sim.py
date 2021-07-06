import numpy as np

from copy import deepcopy
from fclsp.thresh.MeansEstimation import ThreshCV, Empirical
from fclsp.bd_thresh.MeansEstimation import FclspLLACV
from fclsp.reshaping_utils import vec_hollow_sym, fill_hollow_sym

from repro_lap_reg.toy_data.additive_noise import sample_add_noise
from repro_lap_reg.means_est_results import get_means_est_results
from repro_lap_reg.toy_data.adj_mats import block_diag_complete
from repro_lap_reg.utils import merge_dicts
from repro_lap_reg.sim import run_sim
from repro_lap_reg.sim_from_args import get_block_sizes_from_config, \
    get_pen_kws_from_config


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

    true_means = block_diag_complete(block_sizes=block_sizes)

    X = sample_add_noise(means=vec_hollow_sym(true_means),
                         scale=args.noise_scale,
                         n_samples=args.n_samples,
                         random_state=args.data_seed)

    # setup true means and sample data
    # data_sampler = get_add_noise_sampler(means=vec_hollow_sym(true_means),
    #                                      scale=args.noise_scale,
    #                                      noise_dist='gauss')
    # X = data_sampler(n_samples=args.n_samples, random_state=args.data_seed)

    ###############
    # initializer #
    ###############

    default_init_est = ThreshCV(thresh='hard', **gen_thresh_kws)
    default_init_est.set_params(cv=10)  # always make this 10
    default_init_est.fit(X)

    empir_init_est = Empirical()
    empir_init_est.fit(X)

    ################
    # setup models #
    ################

    zero_init = {'values': np.zeros_like(vec_hollow_sym(true_means))}

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
    models['empirical'] = Empirical()

    # Genralized thresholding models
    for kind in ['hard', 'soft']:  # 'adpt_lasso',
        models['thresh__kind={}'.format(kind)] = \
            ThreshCV(thresh=kind, **gen_thresh_kws)

    # FLCP models
    for (stub, strat_kws) in lla_strategies:

        # set initialization
        if strat_kws['init'] == 'default':
            strat_kws['init'] = default_init_est

        elif strat_kws['init'] == 'empirical':
            strat_kws['init'] = empir_init_est

        elif strat_kws['init'] == 'zero':
            strat_kws['init'] = zero_init

        models['fclsp__{}'.format(stub)] = \
            FclspLLACV(**merge_dicts(fclsp_kws, strat_kws))

    out, models = single_sim_from_data(X=X, models=models,
                                       true_means=true_means,
                                       verbosity=verbosity,
                                       zero_tol=args.results_zero_tol)

    out['args'] = args

    return out, models


def single_sim_from_data(X, models, true_means, verbosity=1,
                         zero_tol=1e-8):
    """
    Runs a single simulation on a dataset.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The dataset.

    models: dict of estimators.
        The competing models we want to fit.

    true_means: array-like, (n_features, n_features)
        The true means parameter as an adjacency matrix.

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

    return run_sim(get_est_results=get_means_est_results,
                   get_tune_path_info=get_tune_path_info,
                   get_oracle_est=get_means_oracle_est,
                   models=models,
                   true_param=true_means,
                   X=X,
                   y=None,
                   verbosity=verbosity,
                   zero_tol=zero_tol)


def get_means_oracle_est(true_support, X, y=None):
    """
    Gets the fitted oracle estimator and block diagonal oracle estimate.

    Parameters
    ----------
    true_support: array-like, shape (n_features, n_features)
        The mask of non-zero entries of the true parameter adjacency matrix.

    X: array-like, shape (n_samples, n_features)
        The training data.

    Output
    ------
    oracle_means, oracle_model

    oracle_means: array-like, shape (n_features, )
        The oracle means estimate.

    oracle_model: estimator
        The fitted oracle estimator.
    """

    # Oracle
    oracle_model = Empirical().fit(X)
    oracle_means_mat = fill_hollow_sym(oracle_model.means_)
    oracle_means_mat[~true_support] = 0
    oracle_model.means_ = vec_hollow_sym(oracle_means_mat)

    return oracle_means_mat, oracle_model


def get_tune_path_info(est):
    """
    Gets the tuning path info for models with tuning paths. If the model does not have a tuning path, will return None.

    Parameters
    ----------
    est: Estimator()
        The top level estimator e.g. the cross-validation estimator.

    Output
    ------
    base: Estimator()
        The base estimator to be fit.

    param_seq: list of floats
        The tuning parameter sequence.

    param_name: str
        Name of the tuning parameter.
    """

    if hasattr(est, 'best_estimator_'):
        return get_pen_path_setup(est)
    else:
        return None, None, None


def get_pen_path_setup(est_cv):
    """
    Gets the tuning path information for a model with a tuning parameter.
    """
    param_name = 'pen_val'
    param_seq = est_cv.pen_val_seq_
    base = deepcopy(est_cv.best_estimator_)
    return base, param_seq, param_name
