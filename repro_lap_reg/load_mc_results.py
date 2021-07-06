from os.path import join
import os
from joblib import load
import pandas as pd
from warnings import warn

from repro_lap_reg.results_parsing import ensure_numeric, \
    parse_name, list_expers_with_param  # get_path_best_vs_truth


def load_mc_results(base_name, out_data_dir, handle_missing='warn'):
    """
    Loads the monte-carlo results for a simulation and does some minor formatting.

    Parameters
    ----------
    base_name: str
        Base name of the experiment whose results we are loading.

    out_data_dir: str
        Directory where the experiments are saved.

    handle_missing: str
        How to hanle missing results files. Must be one of ['warn', 'error']
    """

    mc_out = load_mc_out_data(base_name, out_data_dir)
    n_mc = len(mc_out)

    # fit results
    fit_results = concat_mc_dfs([mc_out[mc_idx]['fit']['results']
                                 for mc_idx in range(n_mc)])

    tune_param_names = mc_out[0]['path']['param_name']

    # fit_runtimes = concat_mc_dfs([mc_out[mc_idx]['fit']['runtimes']
    #                              for mc_idx in range(n_mc)])

    # best_path_results = concat_mc_dfs([get_path_best_vs_truth(mc_out[mc_idx])
    #                                    for mc_idx in range(n_mc)])
    path_results = concat_mc_dfs([mc_out[mc_idx]['path']['results']
                                  for mc_idx in range(n_mc)])

    return {'fit': fit_results,
            'path': path_results,
            'tune_param_names': tune_param_names,
            # 'fit_runtimes': fit_runtimes
            }


def concat_mc_dfs(dfs):
    n_mc = len(dfs)
    for mc_idx in range(n_mc):
        dfs[mc_idx]['mc_idx'] = mc_idx

    dfs = pd.concat(dfs).reset_index(drop=True)
    dfs.insert(0, 'mc_idx', dfs.pop('mc_idx'))  # make mc_idx the first column

    # for some crazy reason pandas is not properly detectings flaots
    dfs = ensure_numeric(dfs)
    dfs['mc_idx'] = dfs['mc_idx'].astype(int)
    return dfs


def load_mc_out_data(base_name, out_data_dir, handle_missing='warn'):
    """
    Loads the Monte-Carlo results.

    Parameters
    ----------
    base_name: str
        Base name of the experiment whose results we are loading.

    out_data_dir: str
        Directory where the experiments are saved.

    handle_missing: str
        How to hanle missing results files. Must be one of ['warn', 'error']

    Output
    ------
    out: list of dicts
    """
    assert handle_missing in ['warn', 'error']

    # get experiments correspondint to this base name
    expers = list_expers_with_param(base_name=base_name, param='mc',
                                    out_data_dir=out_data_dir)
    assert len(expers) >= 1

    mc_out = [None for _ in range(len(expers))]
    print('Loading monte-carlo experiments from {}'.format(base_name))
    print(expers)

    for name in expers:

        # get monte-carlo index
        metadata = parse_name(name)
        mc_idx = int(metadata['mc'])

        # make sure there are no repeated monte-carlo indices
        assert mc_out[mc_idx] is None

        # load results
        results_fpath = join(out_data_dir, name, 'results')
        if os.path.exists(results_fpath):
            mc_out[mc_idx] = load(results_fpath)

        elif handle_missing == 'warn':
            warn("Missing results file {}".format(results_fpath))

        elif handle_missing == 'error':
            raise ValueError("Missing results file {}".format(results_fpath))

    # get rid of the Nones for missing results files
    mc_out = [x for x in mc_out if x is not None]
    assert len(mc_out) >= 1

    return mc_out
