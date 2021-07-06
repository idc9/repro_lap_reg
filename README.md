The code in this repository reproduces the simulation results from [The folded concave Laplacian spectral penalty learns block diagonal sparsity patterns with the strong oracle property](TODO: arxiv link).



# Reproduce simulation results


```
# create a conda environment (optional)
conda create -n repro_lap_reg python=3.6
conda activate repro_lap_reg

# download code 
git clone https://github.com/idc9/repro_lap_reg@v0.0.0
pip install .

# modify the paths in run_all_simulations.sh

# run all simulations!
sh run_all_simulations.sh
```
The individual experiments range between between a minute and ~8 hours to run. If you have access to a computing cluster you may want to parallelize the simulations, which should require minimal changes to the current code.


# Notes about code organization

We have tried to make the code modular so that you can swap in your favorite dataset example or add in a new competing model.

### Simulation modules

The sim.py module is the core simulation script for all models. This takes a sampled dataset, fits various competing models then computes a variety of performance measures for each model. The scripts named MODEL_sim.py take care of details for each model e.g. sampling the linear regression data and call the sim.py module.

### Simulations scripts

The scripts/ folder has scripts named run_one_MODEL_sim.py and submit_MODEL_sims.py. The former runs a single simulation meaning it samples a single dataset then fits all competing models. The latter submits multiples simulations for different numbers of samples, different block sizes and multiple monte-carlo repetitions. 

The script viz_one_sim.py makes detailed output for a single simulation. The script viz_seq_sim.py makes detailed output for a sequence of number of samples with multiple monte-carlo repititions. viz_for_paper.py is similar, but makes nicer plots.


### Bayes computing cluster
The original experiments were performed on UW's computing cluster and the code in submit_to_cluster/ handles submitting experiments on this cluster. The code should run just find on your laptop as is, but you will need to modify it if you want to run it on your own computing cluster.
