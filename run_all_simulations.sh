#!/bin/sh


# This script runs all the simulations from (Carmichael, 2021). It assumes you first run git clone www.github.com/idc9/repro_lap_reg and cd into the cloned directory.


# Change these to match your setup!!
script_dir=/Users/iaincarmichael/Dropbox/Research/local_packages/python/repro_lap_reg/scripts/
out_data_dir=/Users/iaincarmichael/Dropbox/Research/laplacian_reg/sim/out_data/
results_dir=/Users/iaincarmichael/Dropbox/Research/laplacian_reg/sim/results/

#########
# setup #
#########


# install required packages
pip install -r requirements.txt

# Manually install a couple packages
pip install git+https://github.com/idc9/ya_glm.git@v0.0.0
pip install git+https://github.com/idc9/fclsp.git@v0.0.0

# pip install git+https://github.com/mathurinm/andersoncd.git
cd andersoncd
pip install .
cd ..



###################
# Run simulations #
###################
python scripts/submit_means_est_sims.py --n_mc_reps 20 --out_data_dir $out_data_dir --results_dir $results_dir --script_dir $script_dir
python scripts/submit_covar_sims.py --n_mc_reps 20 --out_data_dir $out_data_dir --results_dir $results_dir --script_dir $script_dir
python scripts/submit_lin_reg_sims.py --n_mc_reps 20 --out_data_dir $out_data_dir --results_dir $results_dir --script_dir $script_dir
python scripts/submit_log_reg_sims.py --n_mc_reps 20 --out_data_dir $out_data_dir --results_dir $results_dir --script_dir $script_dir

#######################
# Make visualizations #
#######################
python scripts/submit_means_est_sims.py --make_viz --out_data_dir $out_data_dir --results_dir $results_dir --script_dir $script_dir
python scripts/submit_covar_sims.py --make_viz --out_data_dir $out_data_dir --results_dir $results_dir --script_dir $script_dir
python scripts/submit_lin_reg_sims.py --make_viz --out_data_dir $out_data_dir --results_dir $results_dir --script_dir $script_dir
python scripts/submit_log_reg_sims.py --make_viz --out_data_dir $out_data_dir --results_dir $results_dir --script_dir $script_dir
