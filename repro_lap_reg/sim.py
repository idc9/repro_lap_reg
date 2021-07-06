from time import time
import pandas as pd
import numpy as np
from copy import deepcopy

from repro_lap_reg.utils import get_datetime, is_power_of_2


# TODO: possibly give option to pass in test data
def run_sim(get_est_results,
            get_tune_path_info,
            get_oracle_est,
            models, true_param,
            X, y=None,
            verbosity=1,
            zero_tol=1e-8):

    """

    Parameters
    ----------
    get_est_results: callable(est, true_param, zero_tol) -> dict

    get_tune_path_info: callable(est) -> base, param_seq, param_name

    oracle_getter: callabel(X, y, true_support) -> oracle_param, oracle_model

    models: dict of estimators

    true_param: array-like
        The true parameter.

    X: array-like, shape (n_samples, n_features)
        The training X data.

    y: None, array-like
        The training y data.

    verbose: int
        How much printout we want.

    zero_tol: float

    Output
    ------
    results, models

    results: dict
        results['fit']
        results['path']
        results['sim_runtime']
        results['true_param']
        results['oracle_param']

    models: dict
        models['fit']
        models['best_path']

    """

    sim_datetime = get_datetime()
    sim_start_time = time()

    if verbosity >= 1:
        print("starting simulation at {}".format(sim_datetime))

    ########################
    # Fit models from data #
    ########################
    # e.g. select tuning parameters with cross-validation

    fit_vs_true = {}
    fit_vs_oracle = {}
    fit_runtimes = {}

    # get support of true parameter
    true_support = abs(true_param) > zero_tol

    # fit oracle estimator
    start_time = time()
    oracle_param, models['oracle'] = \
        get_oracle_est(X=X, y=y, true_support=true_support)
    fit_runtimes['oracle'] = time() - start_time

    # fit each model
    for name in models.keys():

        # fit model
        if name not in ['oracle', 'init']:  # we already fit these!
            start_time = time()
            models[name].fit(X=X, y=y)
            fit_runtimes[name] = time() - start_time

        # compare estimate to true parameter
        fit_vs_true[name] = get_est_results(est=models[name],
                                            true=true_param,
                                            zero_tol=zero_tol)

        # compare estimate to oracle estimator
        if name != 'oracle':
            fit_vs_oracle[name] = get_est_results(est=models[name],
                                                  true=oracle_param,
                                                  zero_tol=zero_tol)

        # maybe print simulation progress
        if verbosity >= 1:
            print('{} fitted after {:1.2f} seconds with L2 error {:1.4f}'.
                  format(name,
                         fit_runtimes[name],
                         fit_vs_true[name]['L2']))

        # TODO: add runtimes to results e.g.
        fit_vs_true[name]['runtime'] = fit_runtimes[name]
        # fit_vs_oracle[name]['runtime'] = np.nan  # fit_runtimes[name]

    # Format as data frames
    fit_vs_true = pd.DataFrame(fit_vs_true).T
    fit_vs_true.index.name = 'model'
    fit_vs_true['vs'] = 'truth'

    fit_vs_oracle = pd.DataFrame(fit_vs_oracle).T
    fit_vs_oracle.index.name = 'model'
    fit_vs_oracle['vs'] = 'oracle'

    fit_results = pd.concat([fit_vs_true.reset_index(),
                             fit_vs_oracle.reset_index()])

    # fit_runtimes = pd.Series(fit_runtimes, name='runtime')
    # fit_runtimes.index.name = 'model'

    #####################
    # best in full path #
    #####################

    path_results = {'param_name': {}, 'param_seq': {},
                    'best_idx': {}}
    path_vs_true = []
    path_vs_oracle = []
    path_runtimes = []

    path_best_models = {}
    for name in models.keys():

        # get tuning information for models with a tuning path
        base, param_seq, param_name = get_tune_path_info(models[name])

        # if this model is not tuned over a path then skip it
        if base is None:
            continue

        # format parameter sequence
        param_seq = pd.Series(param_seq, name=param_name)
        param_seq.index.name = 'path_idx'
        path_results['param_seq'][name] = param_seq
        path_results['param_name'][name] = param_name

        vs_true = []
        vs_oracle = []
        runtimes = []

        best_L2_vs_true = np.inf

        # estimator_path = []
        for i, param_val in enumerate(param_seq):
            start_time = time()

            # fit model at each parameter setting
            # est = clone(base) # not working with the current version of ya_glm
            est = deepcopy(base)
            est.set_params(**{param_name: param_val})
            est.fit(X=X, y=y)
            runtime = time() - start_time

            # compare this model to both true parameter and oracle estimate
            res_true = get_est_results(est=est,
                                       true=true_param,
                                       zero_tol=zero_tol)

            res_oracle = get_est_results(est=est,
                                         true=oracle_param,
                                         zero_tol=zero_tol)

            res_true['runtime'] = res_oracle['runtime'] = runtime

            # the model in the tuning path whose estimate is
            # closes to the true parameter
            if res_true['L2'] < best_L2_vs_true:
                best_idx = i
                best_estimator = deepcopy(est)
                best_L2_vs_true = deepcopy(res_true['L2'])

            # save results for this parameter setting
            vs_true.append(res_true)
            vs_oracle.append(res_oracle)
            runtimes.append(runtime)

            # maybe print progress
            if verbosity >= 2 and is_power_of_2(i):
                print('{} path ({}/{}) took {:1.5f} seconds to run'.
                      format(name, i, len(param_seq), runtime))

        # format vs true
        vs_true = pd.DataFrame(vs_true)
        vs_true['model'] = name
        vs_true.index.name = 'path_idx'
        path_vs_true.append(vs_true)

        # format oracle
        vs_oracle = pd.DataFrame(vs_oracle)
        vs_oracle['model'] = name
        vs_oracle.index.name = 'path_idx'
        path_vs_oracle.append(vs_oracle)

        # format runtime
        runtimes = pd.Series(runtimes, name='runtime')
        runtimes.index.name = 'path_idx'
        runtimes = pd.DataFrame(runtimes)
        runtimes['model'] = name
        path_runtimes.append(runtimes)

        # index of the tuning parameter whose estimate
        # is closest to the true parameter
        # min_L2_path = vs_true['L2'].min()
        # best_idx = vs_true.query("L2 == @min_L2_path").index.values.min()
        path_results['best_idx'][name] = best_idx
        path_best_models[name] = best_estimator

    def to_df(res):
        res = pd.concat(res).reset_index()
        res.insert(0, 'model', res.pop('model'))  # make model the first column
        return res

    # put oracle and true results together into one data frame
    path_vs_true = to_df(path_vs_true)
    path_vs_true['vs'] = 'truth'
    path_vs_oracle = to_df(path_vs_oracle)
    path_vs_oracle['vs'] = 'oracle'
    path_results['results'] = pd.concat([path_vs_true, path_vs_oracle])

    # make vs the second column
    path_results['results'].insert(1, 'vs',
                                   path_results['results'].pop('vs'))

    path_results['runtimes'] = to_df(path_runtimes)

    # maybe print runtime
    sim_runtime = time() - sim_start_time
    if verbosity >= 1:
        print("Simulation took {:1.2f} seconds".format(sim_runtime))

    out = {'fit': {'results': fit_results,
                   # 'runtimes': fit_runtimes
                   },

           'path': path_results,

           'sim_runtime': sim_runtime,
           'sim_datetime': sim_datetime,

           'true_param': true_param,
           'oracle_param': oracle_param
           }

    return out, {'fit': models, 'best_path': path_best_models}
