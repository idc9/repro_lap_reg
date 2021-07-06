import argparse

from repro_lap_reg.submit_to_cluster.bayes_cluster import add_args_for_bayes
from repro_lap_reg.sim_from_args import add_multiple_submit_config
from repro_lap_reg.sim_script_utils import submit_multiple_sims, \
    get_n_samples_seq

parser = argparse.\
    ArgumentParser(description='Submit all simulations for block diagonal linear regression.')

parser = add_multiple_submit_config(parser)
parser = add_args_for_bayes(parser, force_add=True)

parser.add_argument('--X_dist', default='indep', type=str,
                    choices=['indep', 'corr'],
                    help='X data distribution.')


args = parser.parse_args()

base_kwargs = {'n_pen_vals': 200,
               'X_dist': args.X_dist,
               'X_corr_strength': .1}

base_options = ['add_ridge']

if base_kwargs['X_dist'] == 'indep':
    name_stub = 'X=indep'
elif base_kwargs['X_dist'] == 'corr':
    name_stub = 'X=corr'


for node_size in ['small', 'med', 'large']:

    if node_size == 'small':
        block_size_str_seq = ['5_2']

        n_samples_seq = get_n_samples_seq(start=25, by_small=25,
                                          mid=200, by_big=50, end=500)

    elif node_size == 'med':
        block_size_str_seq = ['10_2']
        n_samples_seq = get_n_samples_seq(start=100, by_small=50,
                                          mid=500, by_big=100, end=1500)

    elif node_size == 'large':
        block_size_str_seq = ['25_2']
        n_samples_seq = get_n_samples_seq(start=500, by_small=250,
                                          mid=3000, by_big=1000, end=8000)

    # format mini experiment
    if args.mini:
        n_samples_seq = n_samples_seq[0:3]
        base_kwargs['n_pen_vals'] = 5
        if node_size != 'small':
            break

    submit_multiple_sims(args=args, model_name='log_reg',
                         base_options=base_options,
                         base_kwargs=base_kwargs,
                         n_samples_seq=n_samples_seq,
                         block_size_str_seq=block_size_str_seq,
                         name_stub=name_stub)
