from joblib import dump
from os.path import join
import os
import numpy as np
from itertools import product
from time import time

from fclsp.reshaping_utils import fill_hollow_sym

from repro_lap_reg.lin_reg_results import get_coef
from repro_lap_reg.covar_results import get_covar
from repro_lap_reg.means_est_results import get_means

from repro_lap_reg.utils import join_and_make
from repro_lap_reg.submit_to_cluster.utils import which_computer
from repro_lap_reg.covar_sim import single_sim_from_args as run_cov_sim
from repro_lap_reg.lin_reg_sim import single_sim_from_args as run_lin_reg_sim
from repro_lap_reg.log_reg_sim import single_sim_from_args as run_log_reg_sim

from repro_lap_reg.means_est_sim import \
    single_sim_from_args as run_means_est_sim

from repro_lap_reg.utils import get_seeds
from repro_lap_reg.submit_utils import get_command_func
from repro_lap_reg.submit_to_cluster.bayes_cluster import get_bayes_command


def run_sim_from_args(args, model_name, verbosity=1):
    """
    Runs a simulation from arguments.
    """

    # format mini experiment
    if args.mini:
        args.n_pen_vals = 5

    if verbosity >= 1:
        print(args)

    ######################
    # setup directories #
    #####################

    script_dir = args.script_dir
    out_data_dir = args.out_data_dir
    save_dir = join_and_make(out_data_dir, model_name, args.name)

    ##################
    # run simulation #
    ##################

    if model_name == 'means_est':
        out, models = run_means_est_sim(args=args, verbosity=np.inf)

    elif model_name == 'covar':
        out, models = run_cov_sim(args=args, verbosity=np.inf)

    elif model_name == 'lin_reg':
        out, models = run_lin_reg_sim(args=args, verbosity=np.inf)

    elif model_name == 'log_reg':
        out, models = run_log_reg_sim(args=args, verbosity=np.inf)

    else:
        raise ValueError("Bad input to model_name: {}".format(model_name))

    # save results
    dump(out, join(save_dir, 'results'))

    # maybe save fit models
    if args.save_models:
        dump(models, join(save_dir, 'models'))

    # ##################################
    # # maybe run visualization script #
    # ##################################
    if args.make_viz:

        # path to vizualzation script
        viz_script_fpath = join(script_dir, 'viz_one_sim.py')

        path_args = '--out_data_dir {} --results_dir {}'.\
            format(args.out_data_dir, args.results_dir)

        command = 'python {} --name {} --kind {} {}'.\
            format(viz_script_fpath, args.name, model_name, path_args)

        os.system(command)


# TODO: where should this go?
def get_param_as_adj(est, kind):

    if kind == 'means_est':
        means = get_means(est)
        return fill_hollow_sym(means)

    elif kind == 'covar':
        return get_covar(est)

    elif kind == 'lin_reg':
        coef = get_coef(est)[0]
        coef = fill_hollow_sym(coef)
        return coef

    elif kind == 'log_reg':
        coef = get_coef(est)[0]
        coef = fill_hollow_sym(coef)
        return coef


def submit_multiple_sims(args, model_name, base_options, base_kwargs,
                         n_samples_seq, block_size_str_seq, name_stub=None):

    script_dir = args.script_dir
    base_kwargs['out_data_dir'] = args.out_data_dir
    base_kwargs['results_dir'] = args.results_dir

    # force mini experiment
    if args.mini:
        # base_options.append('mini')  # now taken care of below
        n_samples_seq = [50, 60]
        block_size_str_seq = ['3_3', '3_2']

        args.n_mc_reps = 2

    ############################
    # Just make visualizations #
    ############################
    # maybe just run the visualization script
    if args.make_viz:
        path_args = '--out_data_dir {} --results_dir {}'.\
                format(args.out_data_dir, args.results_dir)

        ################################
        # make sequence visulaizations #
        ################################
        for viz_script in ['viz_seq_sim', 'viz_for_paper']:

            if name_stub is None:
                command_base = 'python {}.py {} --kind {} '\
                        '--base_name bsize='.format(viz_script, path_args,
                                                    model_name)
            else:
                command_base = 'python {}.py {} --kind {} '\
                        '--base_name {}__bsize='.format(viz_script, path_args,
                                                        model_name,
                                                        name_stub)

            for bsize in block_size_str_seq:
                command = command_base + bsize
                os.system(command)

        # stop here!
        return None

    ###############################
    # Or actually run simulations #
    ###############################

    # setup paths
    script_fpath = join(script_dir, 'run_one_{}_sim.py'.format(model_name))

    # function that returns the run command
    get_command = get_command_func(script_fpath=script_fpath,
                                   base_kwargs=base_kwargs,
                                   base_options=base_options)

    # sample data seeds
    data_seeds = get_seeds(n_seeds=args.n_mc_reps, random_state=args.metaseed)

    ######################
    # submit experiments #
    ######################
    exper_idx = 0
    start_time = time()
    for mc_idx in range(args.n_mc_reps):

        # possibly skip this MC run
        if mc_idx < args.mc_start_idx:
            continue

        for n_samples, block_size_str in product(n_samples_seq,
                                                 block_size_str_seq):
            exper_idx += 1
            add_options = []
            add_kwargs = {}

            ##############################
            # setup experiment arguments #
            ##############################

            add_kwargs['n_samples'] = n_samples
            add_kwargs['block_size_str'] = block_size_str

            # name experiment
            if name_stub is not None:
                name = name_stub + '__'
            else:
                name = ''
            name += 'bsize={}__n_samples={}'.format(block_size_str, n_samples)
            name += '__mc={}'.format(mc_idx)
            add_kwargs['name'] = name

            # seeds for monte-carlo repitition
            add_kwargs['data_seed'] = data_seeds[mc_idx]

            # save and visualize extra information for the first MC repitition
            if mc_idx == 0:
                add_options.append('save_models')
                add_options.append('make_viz')

            # get command to run run_one_MODEL_sim.py
            command = get_command(add_kwargs=add_kwargs,
                                  add_options=add_options)

            # modify command to submit a job on the Bayes cluster
            if which_computer() == 'bayes' and args.submit:
                command = command.split('python ')[1]  # drop the 'python '
                command = get_bayes_command(py_command=command, args=args)

            # run the command
            print('\n\nSubmission command for experiment {}:'.
                  format(exper_idx))
            print(command)
            # if not args.print:
            os.system(command)

    runtime = time() - start_time
    print('{} experiments took {:1.2f} seconds to submit'.
          format(exper_idx, runtime))


def get_n_samples_seq(start=10, by_small=10, mid=100, by_big=50, end=300):
    """
    Returns a sequence where the first set of values is narrowly spaced apart and the second set of values is spaced further apart.

    Parameters
    ----------
    start: int
        Where to start the sequence.

    by_small: int
        How far apart the narrowly spaced values are.

    mid: int
        Where the sequence changes from narrow to large spacing.

    by_big: int
        How far apart the widely spaced values are.

    end: int
        Where to end the sequence.

    Ouput
    -----
    values: np.array
    """
    assert start < mid
    assert mid < end

    return np.concatenate([np.arange(start, mid + 1, by_small),
                           np.arange(mid + by_big, end + 1, by_big)])
