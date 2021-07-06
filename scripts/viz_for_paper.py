from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import argparse

from repro_lap_reg.load_param_seq_results import load_param_seq_results
from repro_lap_reg.viz_seq import plot_param_vs_metric_by_model
from repro_lap_reg.viz_utils import savefig
from repro_lap_reg.utils import join_and_make


parser = argparse.ArgumentParser(description='Visualiztions for experiments'
                                             'with n_samples sequence.')

parser.add_argument('--out_data_dir',
                    default='out_data/',
                    help='Directory for output data.')

parser.add_argument('--results_dir',
                    default='results/',
                    help='Directory where visualizations should be saved.')


parser.add_argument('--kind', default='covar',
                    help='What kind of model are we looking at.')


parser.add_argument('--base_name', default='meow',
                    help='Base name to identify simulations.')

args = parser.parse_args()

# args.kind = 'means_est'
# args.base_name = 'bsize=5_2'
show_std = True


dpi = 100
inches = 12
fontsize = 32  # 22
tick_font_size = 20
plt.rc('legend', fontsize=fontsize)

param = 'n_samples'
param_title = "Number of samples"

metrics = ['L2_rel', 'support_error']
metric_best_ordering = {'L2': 'min',
                        'support_error': 'min',
                        'L2_rel': 'min'}

metric_ylims = {'L2': 0,
                'support_error': (0, 1),
                'L2_rel': (0, 1)}

metric_titles = {'L2_rel': 'Relative L2 error',
                 'L2': 'L2 error',
                 'support_error': 'Support error'}


#########
# paths #
#########
results_dir = args.results_dir
out_data_dir = args.out_data_dir
out_data_dir = join(out_data_dir, args.kind)
save_dir = join_and_make(results_dir, 'for_paper')

name_stub = '{}__{}'.format(args.kind, args.base_name)


def get_block_size_title(name):
    #TODO: this won't work for more sophosticated names
    block_size_str = name.split('bsize=')[1]
    vals = block_size_str.split('_')
    assert len(vals) in [2, 3]
    if len(vals) == 2:
        n_nodes = vals[0]
        n_blocks = vals[1]
        return "{} blocks with {} nodes".format(n_blocks, n_nodes)
    elif len(vals) == 3:
        n_nodes = vals[0]
        n_blocks = vals[1]
        n_iso = vals[2]
        return "{} blocks with {} nodes and {} isolated vertices".\
            format(n_blocks, n_nodes, n_iso)


title_stub = get_block_size_title(args.base_name)


##########################
# load and parse results #
##########################


out = load_param_seq_results(base_name=args.base_name,
                             param=param,
                             out_data_dir=out_data_dir)

fit_results = out['fit']
path_results = out['path']


if args.kind == 'means_est':
    defualt_init_name = '10-CV hard-threshold init'

    all_models = [
        'fclsp__init=default__steps=2',
        'fclsp__init=default__steps=convergence',
        'fclsp__init=0__steps=3',
        'fclsp__init=0__steps=convergence',
        'fclsp__init=empirical__steps=2',
        'fclsp__init=empirical__steps=convergence',
        'thresh__kind=hard',
        #'thresh__kind=soft',
        'empirical'
        ]

    figsize = (inches, 1.5 * inches)

elif args.kind == 'covar':
    defualt_init_name = '10-CV hard-threshold init'

    all_models = [
        'fclsp__init=default__steps=2',
        'thresh__kind=hard',
        # 'thresh__kind=soft'
        'empirical'
        ]
    # all_models = [
    #     'fclsp__init=default__steps=2',
    #     'fclsp__init=default__steps=convergence',
    #     'fclsp__init=0__steps=3',
    #     'fclsp__init=0__steps=convergence',
    #     'fclsp__init=empirical__steps=2',
    #     'fclsp__init=empirical__steps=convergence',
    #     'thresh__kind=hard',
    #     'thresh__kind=soft',
    #     'empirical'
    #     ]

    figsize = (inches, inches)


elif args.kind == 'lin_reg':
    defualt_init_name = '10-CV Lasso init'

    all_models = [
        'fclsp__init=default__steps=2',
        'fcp__init=default__steps=1',
        'lasso'
        ]

    # all_models = [
    #     'fclsp__init=default__steps=2',
    #     'fclsp__init=default__steps=convergence',
    #     'fclsp__init=0__steps=3',
    #     'fclsp__init=0__steps=convergence',
    #     'fclsp__init=empirical__steps=2',
    #     'fclsp__init=empirical__steps=convergence',

    #     'fcp__init=default__steps=1',
    #     'fcp__init=default__steps=convergence',
    #     'fcp__init=0__steps=2',
    #     'fcp__init=0__steps=convergence',
    #     'fcp__init=empirical__steps=1',
    #     'fcp__init=empirical__steps=convergence',

    #     'lasso'
    #     ]

    figsize = (inches, inches)

elif args.kind == 'log_reg':
    defualt_init_name = '10-CV Lasso init'

    all_models = [
        'fclsp__init=default__steps=2',
        'fcp__init=default__steps=1',
        'lasso'
        ]

    # all_models = [
    #     'fclsp__init=default__steps=2',
    #     'fclsp__init=default__steps=convergence',
    #     'fclsp__init=0__steps=3',
    #     'fclsp__init=0__steps=convergence',
    #     'fclsp__init=empirical__steps=2',
    #     'fclsp__init=empirical__steps=convergence',

    #     'fcp__init=default__steps=1',
    #     'fcp__init=default__steps=convergence',
    #     'fcp__init=0__steps=2',
    #     'fcp__init=0__steps=convergence',
    #     'fcp__init=empirical__steps=1',
    #     'fcp__init=empirical__steps=convergence',

    #     'lasso'
    #     ]

    figsize = (inches, inches)

model_color_seq = sns.color_palette(palette='colorblind', n_colors=4)
fclsp_sub_colors = sns.light_palette(color=model_color_seq[0],
                                     n_colors=2+1,
                                     reverse=True)[:-1]

fcp_sub_colors = sns.light_palette(color=model_color_seq[1],
                                   n_colors=2+1,
                                   reverse=True)[:-1]

info = {}

markers = {'fclsp': '.',
           'fcp': 'X',
           '0': '$O$',
           'empirical': '$e$',
           'other': '.'
           }

# FCLSP
info['fclsp__init=default__steps=2'] = {
    'name': 'FCLS, 2 LLA steps,\n    {}'.format(defualt_init_name),
    'color': fclsp_sub_colors[0],
    'ls': '-',
    'marker': markers['fclsp'],
    'path': True
}


info['fclsp__init=default__steps=convergence'] = {
    'name': 'FCLS, LLA converge,\n    {}'.format(defualt_init_name),
    'color': fclsp_sub_colors[1],
    'ls': '-',
    'marker': markers['fclsp'],
    'path': True
}


info['fclsp__init=0__steps=3'] = {
    'name': 'FCLS, 3 LLA steps,\n    init at 0',
    'color': fclsp_sub_colors[0],
    'ls': '-',
    'marker': markers['0'],
    'path': True
}


info['fclsp__init=0__steps=convergence'] = {
    'name': 'FCLS, LLA converge,\n    init at 0',
    'color': fclsp_sub_colors[1],
    'ls': '-',
    'marker': markers['0'],
    'path': True
}


info['fclsp__init=empirical__steps=2'] = {
    'name': 'FCLS, 2 LLA steps,\n    empirical init',
    'color': fclsp_sub_colors[0],
    'ls': '-',
    'marker': markers['empirical'],
    'path': True
}


info['fclsp__init=empirical__steps=convergence'] = {
    'name': 'FCLS, LLA converge,\n    empirical init',
    'color': fclsp_sub_colors[1],
    'ls': '-',
    'marker': markers['empirical'],
    'path': True
}

# FCP
info['fcp__init=default__steps=1'] = {
    'name': 'SCAD, 1 LLA step,\n    {}'.format(defualt_init_name),
    'color': fcp_sub_colors[0],
    'ls': '--',
    'marker': '$S$',  # markers['fcp'],
    'path': True
}


info['fcp__init=default__steps=convergence'] = {
    'name': 'SCAD, LLA converge,\n    {}'.format(defualt_init_name),
    'color': fcp_sub_colors[1],
    'ls': '--',
    'marker': markers['fcp'],
    'path': True
}


info['fcp__init=0__steps=2'] = {
    'name': 'SCAD, 2 LLA steps,\n    init at 0',
    'color': fcp_sub_colors[0],
    'ls': '--',
    'marker': markers['0'],
    'path': True
}


info['fcp__init=0__steps=convergence'] = {
    'name': 'SCAD, LLA converge,\n    init at 0',
    'color': fcp_sub_colors[1],
    'ls': '--',
    'marker': markers['0'],
    'path': True
}


info['fcp__init=empirical__steps=1'] = {
    'name': 'Entrywise SCAD, 1 LLA step,\n    empirical init',
    'color': fcp_sub_colors[0],
    'ls': '--',
    'marker': markers['empirical'],
    'path': True
}


info['fcp__init=empirical__steps=convergence'] = {
    'name': 'SCAD, LLA converge,\n    empirical init',
    'color': fcp_sub_colors[1],
    'ls': '--',
    'marker': markers['empirical'],
    'path': True
}

#########
# Other #
#########

info['lasso'] = {
    'name': 'Lasso',
    'color': model_color_seq[3],
    'ls': '-.',
    'marker': '$L$',
    'path': False
    }

info['thresh__kind=hard'] = {
    'name': 'hard-thresholding',
    'color': fcp_sub_colors[0],
    'ls': '--',
    'marker': '$H$',
    'path': True
    }


info['thresh__kind=soft'] = {
    'name': 'soft-thresholding',
    'color': model_color_seq[3],
    'ls': '-.',
    'marker': markers['other'],
    'path': True
    }

info['empirical'] = {
    'name': 'Empirical',
    'color': model_color_seq[3],
    'ls': '-.',
    'marker': '$E$',  # markers['other'],
    'path': False
    }


for k in list(info.keys()):
    if k not in all_models:
        del info[k]

formal_names = {k: info[k]['name'] for k in info.keys()}
model_ls = {k: info[k]['ls'] for k in info.keys()}
model_markers = {k: info[k]['marker'] for k in info.keys()}
model_colors = {k: info[k]['color'] for k in info.keys()}


path_models = [k for k in info.keys() if info[k]['path']]
fit_models = [k for k in info.keys() if not info[k]['path']]


###############
# make figure #
###############

metric = 'L2_rel'
vs = 'oracle'


# pull out the best result for the tuning path
best_results = path_results.\
    query("vs == @vs").\
    groupby(['model', 'mc_idx', 'n_samples'])[metric].\
    agg(metric_best_ordering[metric]).\
    reset_index()


# get results for models we actually are going to plot
results_to_plot = pd.concat([best_results.query('model in @all_models'),
                             fit_results.query("vs == @vs and model in @fit_models")]).\
                                reset_index()


plt.figure(figsize=figsize)
plot_param_vs_metric_by_model(results=results_to_plot,
                              grp_var=param,
                              metric=metric,
                              colors=model_colors,
                              show_std=show_std,
                              ls=model_ls,
                              markers=model_markers,
                              marker_size=fontsize,
                              label_dict=formal_names)

plt.ylim(metric_ylims[metric])
plt.ylabel('{} to {}'.format(metric_titles[metric], vs),
           fontsize=fontsize)
plt.xlabel(param_title, fontsize=fontsize)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)

plt.title(title_stub, fontsize=fontsize)
savefig(join(save_dir, '{}__vs__{}__{}.png'.format(name_stub, vs, metric)),
        dpi=dpi)
