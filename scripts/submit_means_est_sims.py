import argparse

from repro_lap_reg.submit_to_cluster.bayes_cluster import add_args_for_bayes
from repro_lap_reg.sim_from_args import add_multiple_submit_config
from repro_lap_reg.sim_script_utils import submit_multiple_sims, \
    get_n_samples_seq

parser = argparse.\
    ArgumentParser(description='Submit all simulations for means estimation with block diagiaonal shrinkage.')
parser = add_multiple_submit_config(parser)
parser = add_args_for_bayes(parser, force_add=True)


args = parser.parse_args()

base_kwargs = {'n_pen_vals': 200,
               'noise_scale': 3.25
               }


block_size_str_seq = ['5_2', '10_2', '25_2']


n_samples_seq = get_n_samples_seq(start=10, by_small=10,
                                  mid=100, by_big=50, end=600)


# format mini experiment
if args.mini:
    n_samples_seq = n_samples_seq[0:3]
    base_kwargs['n_pen_vals'] = 5
    block_size_str_seq = ['5_2']

submit_multiple_sims(args=args, model_name='means_est',
                     base_options=[],
                     base_kwargs=base_kwargs,
                     n_samples_seq=n_samples_seq,
                     block_size_str_seq=block_size_str_seq,
                     name_stub=None)
