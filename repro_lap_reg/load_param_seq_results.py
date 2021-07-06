import pandas as pd

from repro_lap_reg.results_parsing import parse_name, list_expers_with_param
from repro_lap_reg.load_mc_results import load_mc_results


def load_param_seq_results(base_name, param, out_data_dir,
                           handle_missing='warn'):
    """
    Loads the results for an experiment across a sequence of parameter values e.g. n_samples.

    Parameters
    ----------
    base_name: str
        Base name of the experiment whose results we are loading.

    param: str

    out_data_dir: str
        Directory where the experiments are saved.

    handle_missing: str
        How to hanle missing results files. Must be one of ['warn', 'error']

    # TODO: make sure each param value has all the mc iterations
    # TODO: make work for non-float parameters

    Output
    ------
    out: list of dicts
    """

    # get experiments correspondint to this base name
    expers = list_expers_with_param(base_name=base_name, param=param,
                                    out_data_dir=out_data_dir)
    assert len(expers) >= 1

    print('Loading {} data for experiments:'.format(param))
    print(expers)

    # format metadata
    metadata = [parse_name(name) for name in expers]
    assert 'mc' in metadata[0].keys()  # make sure we have monte-carlo repititions
    info = pd.DataFrame({param: [md[param] for md in metadata],
                         'mc_idx': [int(md['mc']) for md in metadata],
                         'exper': expers})

    # make sure we have unique param-MC parings
    assert info[[param, 'mc_idx']].duplicated().sum() == 0

    # make sure every parameter setting has all monte-carlo values
    p0 = info[param].values[0]
    n_mc = info.query("{} == @p0".format(param)).shape[1]
    for p in set(info[param]):
        _n_mc = info.query("{} == @p".format(param)).shape[1]
        assert n_mc == _n_mc

    # sort by experiment values
    info[param] = info[param].astype(float)
    info = info.sort_values(by=[param, 'mc_idx']).reset_index(drop=True)

    print('\n\nDetected the following results')
    print(info)
    print('\n\n')

    # load MC results for each parameter setting
    results = {'fit': [],
               'path': [],
               # 'fit_runtimes': [],
               'param_seq': [],
               'param_name': param}
    for _, row in info.query("mc_idx == 0").iterrows():

        # get name without monte-carlo index
        base_name = row['exper']
        mc_idx = parse_name(base_name)['mc']
        base_name = base_name.replace('__mc={}'.format(mc_idx), '')

        # add tuning parameter sequence
        results['param_seq'].append(row[param])

        # load all monte-carlo sims for this param value
        mc_res = load_mc_results(base_name=base_name,
                                 out_data_dir=out_data_dir,
                                 handle_missing=handle_missing)

        results['fit'].append(mc_res['fit'])
        results['path'].append(mc_res['path'])

    results['tune_param_names'] = mc_res['tune_param_names']

    # concat dfs from each param value
    results['fit'] = concat_dfs(dfs=results['fit'],
                                param_seq=results['param_seq'],
                                param_name=param)

    results['path'] = concat_dfs(dfs=results['path'],
                                 param_seq=results['param_seq'],
                                 param_name=param)
    return results


def concat_dfs(dfs, param_seq, param_name):
    assert len(dfs) == len(param_seq)

    for idx in range(len(dfs)):
        dfs[idx][param_name] = param_seq[idx]

    dfs = pd.concat(dfs).reset_index(drop=True)

    # make param_name the first column
    dfs.insert(0, param_name, dfs.pop(param_name))

    return dfs
