from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

import argparse

from repro_lap_reg.load_param_seq_results import load_param_seq_results
from repro_lap_reg.viz_seq import plot_param_vs_metric_by_model, \
    get_model_colors, get_mc_results_summary_table,\
    get_param_vs_metric_model_ratio_table
from repro_lap_reg.viz_utils import savefig
from repro_lap_reg.utils import join_and_make


parser = argparse.ArgumentParser(description='Visualiztions for experiments'
                                             'with n_samples sequence.')

parser.add_argument('--out_data_dir',
                    default='out_data',
                    help='Directory for output data.')

parser.add_argument('--results_dir',
                    default='results',
                    help='Directory where visualizations should be saved.')


parser.add_argument('--kind', default='covar',
                    help='What kind of model are we looking at.')


parser.add_argument('--base_name', default='meow',
                    help='Base name to identify simulations.')


parser.add_argument('--show_std', action='store_true', default=False,
                    help='Add standard deviation to plots.')

args = parser.parse_args()

# args.show_std = True


dpi = 100
inches = 10
label_font_size = 18
tick_font_size = 12
param = 'n_samples'
param_title = "Number of samples"

metrics = ['L2_rel', 'L2', 'support_error', 'runtime']
metric_best_ordering = {'L2': 'min',
                        'support_error': 'min',
                        'L2_rel': 'min'}

metric_ylims = {'L2': 0,
                'support_error': (0, 1),
                'L2_rel': 0,
                'runtime': 0}

metric_titles = {'L2_rel': 'Relative L2 error',
                 'L2': 'L2 error',
                 'support_error': 'Support error',
                 'runtime': 'Runtime (s)'}


#########
# paths #
#########
results_dir = args.results_dir
out_data_dir = args.out_data_dir
out_data_dir = join(out_data_dir, args.kind)

save_dir = join_and_make(results_dir, args.kind, args.base_name)
table_dir = join_and_make(save_dir, 'tables')


##########################
# load and parse results #
##########################

out = load_param_seq_results(base_name=args.base_name,
                             param=param,
                             out_data_dir=out_data_dir)

fit_results = out['fit']
path_results = out['path']


all_models = np.sort(np.unique(out['fit']['model']))
model_colors = get_model_colors(all_models, cat_palette='husl')

###########################################
# make param seq vs metric visualizations #
###########################################

for metric, vs in product(metrics, ['truth', 'oracle']):

    if metrics == 'runtime' and vs != 'truth':
        continue

    ##########
    # fit vs.#
    ##########
    fit_res = fit_results.query("vs == @vs")

    # make plot
    plt.figure(figsize=(inches, inches))
    plot_param_vs_metric_by_model(results=fit_res,
                                  grp_var=param,
                                  metric=metric,
                                  show_std=args.show_std,
                                  colors=model_colors)
    plt.ylim(metric_ylims[metric])
    plt.ylabel(metric_titles[metric], fontsize=label_font_size)
    plt.xlabel(param_title, fontsize=label_font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)

    savefig(join(save_dir, 'fit_vs_{}_{}.png'.format(vs, metric)),
            dpi=dpi)

    # make table
    get_mc_results_summary_table(results=fit_results,
                                 group_var=param,
                                 metric=metric,
                                 values='formatted').\
        to_csv(join(table_dir, 'fit_vs_{}_{}.csv'.format(vs, metric)))

    if metric == 'runtime':
        continue
    #############
    # best path #
    #############

    # pull out the best result for the tuning path
    best_results = path_results.\
        query("vs == @vs").\
        groupby(['model', 'mc_idx', 'n_samples'])[metric].\
        agg(metric_best_ordering[metric]).\
        reset_index()

    # make plot
    plt.figure(figsize=(inches, inches))
    plot_param_vs_metric_by_model(results=best_results,
                                  grp_var=param,
                                  metric=metric,
                                  show_std=args.show_std,
                                  colors=model_colors)

    plt.ylim(metric_ylims[metric])
    plt.ylabel(metric_titles[metric], fontsize=label_font_size)
    plt.xlabel(param_title, fontsize=label_font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)

    savefig(join(save_dir, 'best_path_vs_{}_{}.png'.format(vs, metric)),
            dpi=dpi)

    # make table
    get_mc_results_summary_table(results=best_results,
                                 group_var=param,
                                 metric=metric,
                                 values='formatted').\
        to_csv(join(table_dir, 'best_path_vs_{}_{}.csv'.format(vs, metric)))

###########################################
# ratio between FCLSP and competing model #
###########################################

model_bot = 'fclsp__init=default__steps=2'
if args.kind in ['covar', 'means_est']:
    model_top = 'thresh__kind=hard'

elif args.kind in ['lin_reg', 'log_reg']:
    model_top = 'fcp__init=default__steps=1'

vs = 'oracle'
metric = 'L2_rel'

# pull out the best result for the tuning path
best_results = path_results.\
    query("vs == @vs").\
    groupby(['model', 'mc_idx', 'n_samples'])[metric].\
    agg(metric_best_ordering[metric]).\
    reset_index()


get_param_vs_metric_model_ratio_table(results=best_results,
                                      grp_var=param,
                                      metric=metric,
                                      model_top=model_top,
                                      model_bot=model_bot).\
    to_csv(join(table_dir,
                'ratio_best_path_vs_{}_{}.csv'.format(vs, metric)))

######################
# Monte-Carlo counts #
######################
mc_counts = fit_results.\
    query("vs == 'truth'").\
    groupby(['model', 'n_samples']).\
    count()['mc_idx'].\
    reset_index()

mc_counts.to_csv(join(table_dir, 'fit_model_mc_counts.csv'))
if len(np.unique(mc_counts['mc_idx'])) > 1:
    print("WARNING: differning numbers of monte-carlo runs detected")
    print(np.unique(mc_counts['mc_idx']))
    print(mc_counts['mc_idx'])
else:
    print("All fit models have same number of monte-carlo runs")
