from os.path import join, exists
from joblib import load
import numpy as np
# import matplotlib.pyplot as plt
import argparse
from itertools import product
import sys

from fclsp.reshaping_utils import fill_hollow_sym

from repro_lap_reg.viz_fit import viz_path_diagnostics, heatmap,\
    print_estimators
# from repro_lap_reg.results_parsing import get_path_best_vs_truth
from repro_lap_reg.viz_utils import savefig
from repro_lap_reg.utils import join_and_make
from repro_lap_reg.sim_script_utils import get_param_as_adj
from repro_lap_reg.ResultsWriter import ResultsWriter

parser = argparse.\
    ArgumentParser(description='Visualize results for one simulation.')

parser.add_argument('--name', default='meow',
                    help='File name stub.')

parser.add_argument('--kind', default='covar',
                    help='What kind of model are we looking at.')

parser.add_argument('--print_only', action='store_true', default=False,
                    help='Show print out results only.')

parser.add_argument('--out_data_dir',
                    default='out_data/',
                    help='Directory for output data.')

parser.add_argument('--results_dir',
                    default='results/',
                    help='Directory where visualizations should be saved.')

args = parser.parse_args()

dpi = 100

#########
# paths #
#########
results_dir = args.results_dir
out_data_dir = args.out_data_dir

results_fpath = join(out_data_dir, args.kind, args.name, 'results')
model_fpath = join(out_data_dir, args.kind, args.name, 'models')

save_dir = join_and_make(results_dir, args.kind, args.name)

################
# load results #
################

# load results output
out = load(results_fpath)

# maybe load saved modes
if exists(model_fpath):
    models = load(model_fpath)
else:
    models = None

# results for the models their best tuning path values
# path_best_vs_truth = get_path_best_vs_truth(out)

# get the true parameter we are targeting
if args.kind == 'covar':
    true_param_adj = out['true_param']
elif args.kind in ['lin_reg', 'log_reg']:
    true_param_adj = fill_hollow_sym(out['true_param'])
elif args.kind == 'means_est':
    true_param_adj = out['true_param']  # fill_hollow_sym(out['true_param'])

##################
# Visualizations #
##################

# create log
writer = ResultsWriter(fpath=join(save_dir, 'log.txt'))
writer.write(out['args'], newlines=1)
writer.write('Simulation ran at {} and took {:1.2f} seconds'.
             format(out['sim_datetime'], out['sim_runtime']), newlines=3)

# print models
if models is not None:
    print_estimators(estimators=models['fit'], print_func=writer.write)

# save fit runtimes data frame
out['fit']['results'].\
    query("vs == 'truth'").\
    set_index('model')['runtime'].\
    sort_values().\
    to_csv(join(save_dir, 'fit_runtimes.csv'), float_format='%1.3f')

# Error metrics for selected models and best path models
for metric, vs in product(['L2', 'support_error'],
                          ['truth', 'oracle']):

    # cross-validation selection results vs true parameter
    out['fit']['results'].\
        query("vs == @vs")[['model', metric]].\
        set_index('model').\
        sort_values(metric).\
        astype(float).\
        to_csv(join(save_dir, 'vs_{}_fit_{}.csv'. format(vs, metric)),
               float_format='%1.4f')

    # get results for best parameter in tuning path
    # path_best_vs_truth.\
    #     query("vs == @vs")[['model', metric]].\
    #     set_index('model').\
    #     sort_values(metric).\
    #     astype(float).\
    #     to_csv(join(save_dir, 'vs_{}_best_path_{}.csv'. format(vs, metric)),
    #            float_format='%1.4f')
    out['path']['results'].\
        query("vs == @vs").\
        groupby('model')[metric].\
        min().\
        sort_values().\
        astype(float).\
        to_csv(join(save_dir, 'vs_{}_best_path_{}.csv'. format(vs, metric)),
               float_format='%1.4f')

if args.print_only:
    sys.exit()

# plot visual diagonstics for models with tuning path
for model_name in out['path']['param_seq'].keys():

    model_dir = join_and_make(save_dir, model_name)

    viz_path_diagnostics(out=out, models=models, model_name=model_name,
                         save_dir=model_dir)


# summarize path runtimes
res = out['path']['results'].query("vs == 'truth'")
path_runtime_summaries = res.\
    groupby('model')['runtime'].\
    agg(**{'mean': np.mean,
           'median': np.median,
           'std': np.std,
           'min': np.min,
           'max': np.max}).\
    sort_values("mean")

path_runtime_summaries.to_csv(join(save_dir, 'path_runtime_summary.csv'))

#################################################
# Heatmaps of the true and estimated parameters #
#################################################
heatmap(true_param_adj)
savefig(join(save_dir, 'true.png'), dpi=dpi)

if models is not None:
    # estimate from cv-fit
    for model_name, model in models['fit'].items():
        model_dir = join_and_make(save_dir, model_name)

        heatmap(get_param_as_adj(model, kind=args.kind))
        savefig(join(model_dir, 'fit.png'), dpi=dpi)

    # cestimate from best path
    for model_name, model in models['best_path'].items():
        model_dir = join_and_make(save_dir, model_name)

        heatmap(get_param_as_adj(model, kind=args.kind))
        savefig(join(model_dir, 'best_path.png'), dpi=dpi)
