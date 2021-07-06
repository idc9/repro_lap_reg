import numpy as np
import argparse

from repro_lap_reg.submit_to_cluster.bayes_cluster import add_args_for_bayes, \
    maybe_submit_on_bayes
from repro_lap_reg.sim_from_args import add_one_sim_config
from repro_lap_reg.sim_script_utils import run_sim_from_args


# setup configuration
parser = argparse.ArgumentParser(description='Run a single simulation for covariance estimation with block diagonal shrinkage.')
parser = add_one_sim_config(parser)
parser = add_args_for_bayes(parser)

parser.add_argument('--support_scale', default=.5, type=float,
                    help='Scale of the non-zero entries of the covariance matrix.')

args = parser.parse_args()
maybe_submit_on_bayes(args)

# run the simulation!
run_sim_from_args(args, model_name='covar', verbosity=np.inf)
