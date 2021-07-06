import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from adjustText import adjust_text
import pandas as pd

from repro_lap_reg.viz_utils import hierarchical_color_palette
from repro_lap_reg.results_parsing import get_model_info


def plot_param_vs_metric_by_model(results, grp_var, metric,
                                  colors='tab10',
                                  ls=None,
                                  markers='.',
                                  marker_size=None,
                                  show_std=True,
                                  add_label=True,
                                  direct_label=False,
                                  label_dict=None):
    """

    Parameters
    ----------
    results: pd.DataFrame

    grp_var: str
        Variable to group by e.g. n_samples.

    metic: str
        Name of the column of results to plot on the yaxis.

    colors: str, dict

    ls: None, str, dict
        Line style

    show_std: bool
        Show the standard deviation.

    add_label: bool
        Add labels to plot.

    direct_label: bool
        Write label directly on each curve.
    """

    all_models = list(set(results['model'].values))
    all_models = np.sort(all_models)  # sort models alphabetically

    def check_input_dict(d):
        assert isinstance(d, dict)
        assert all(m in d.keys() for m in all_models)

    # colors for each model
    if isinstance(colors, str):
        # colors = sns.color_palette(colors, len(all_models))
        # colors = {m: colors[i] for i, m in enumerate(all_models)}
        colors = get_model_colors(model_list=all_models,
                                  cat_palette=colors, light=True)
    else:
        check_input_dict(colors)

    # line style for each model
    if ls is None or isinstance(ls, str):
        ls = {m: ls for m in all_models}
    else:
        check_input_dict(ls)

    # marker style for each model
    if markers is None or isinstance(markers, str):
        markers = {m: markers for m in all_models}
    else:
        check_input_dict(markers)

    # aggregate results
    avg = results.groupby(['model', grp_var])[metric].mean().reset_index()
    # std = results.groupby(['model', grp_var])[metric].std().reset_index()

    std = results.groupby(['model', grp_var])[metric].\
        agg(std_root_n).reset_index()
    # TODO: check we have unique value for each grp_var value

    # make plot for each model
    for m in all_models:
        model_avg = avg.query('model == @m')
        model_std = std.query('model == @m')

        if add_label or not direct_label:
            if label_dict is not None:
                label = label_dict[m]
            else:
                label = m

        else:
            label = None

        plt.plot(model_avg[grp_var], model_avg[metric],
                 marker=markers[m],
                 color=colors[m],
                 ls=ls[m],
                 ms=marker_size,
                 label=label)

        if show_std:
            plt.fill_between(x=model_avg[grp_var],
                             y1=model_avg[metric] + model_std[metric],
                             y2=model_avg[metric] - model_std[metric],
                             color=colors[m],
                             alpha=.4)

    # write label directly on each curve
    if direct_label:

        # labels = []

        for m in all_models:
            model_avg = avg.query('model == @m')

            x = max(model_avg[grp_var])
            y = model_avg.query("{} == @x".format(grp_var))[metric].values[0]

            plt.text(x=x, y=y, s=m,
                     ha='left', va='center', color=colors[m])

            # labels.append(plt.text(x=x, y=y, s=m,
            #                        ha='center', va='center', color=colors[m])
            #                        # fontsize=12)
            #               )

        # adjust_text(labels,
        #             arrowprops=dict(arrowstyle='->', color='black'))

    if add_label and not direct_label:
        plt.legend()

    plt.xlabel(grp_var)
    plt.ylabel(metric)


def get_model_colors(model_list, cat_palette='tab10', light=True):
    """
    Assigns a color to each model in the model list. Makes colors hierarchical by model type.

    Parameters
    ----------
    model_list: list of str

    Output
    ------
    model_colors: dict of colors

    """

    metadata, base_to_model = get_model_info(model_list)

    model_colors, base_colors = \
        hierarchical_color_palette(base_to_model,
                                   cat_palette=cat_palette,
                                   light=light)

    return model_colors


def get_mc_results_summary_table(results, group_var, metric,
                                 values='formatted'):
    """
    Creates the group_var x model table where the ij th entry is the mean, std or mean (std) of the monte-carlo results.

    Parameters
    -----------
    results: pd.DataFrame
        The results for all monte-carlo runs

    group_var: str
        Which column to group by e.g. n_samples.

    metric: str
        Which metric to use e.g. 'L2'

    values: str
        What values to display. Must be one of ['mean', 'std' , 'formatted'].

    Output
    ------
    table: pd.DataFrame
        The aggregated results table.
    """
    assert values in ['mean', 'std', 'formatted']

    results_agg = results.\
        groupby(['model', group_var])[metric].\
        agg(['mean', 'std']).reset_index()
        # agg({'mean': np.mean, 'std': std_root_n}) does this work?
        # TODO

    def format_table(x):
        return '{:1.3f} ({:1.2f})'.format(x['mean'], x['std'])

    results_agg['formatted'] = results_agg.apply(format_table, axis=1)

    piv = results_agg.\
        pivot(index=group_var, columns='model', values=values).\
        sort_index().T.sort_index().T

    return piv


def get_param_vs_metric_model_ratio_table(results, grp_var, metric,
                                          model_top, model_bot):
    """
    Get the ratio between two models over a sequence of values.

    Parameters
    ----------
    results: pd.DataFrame
        The results data frame.

    grp_var: str
        The varialble to group by.

    metric: str
        The metric whose ratio we want to compute.

    model_top: str
        Which model is on top in the ratio.

    model_top: str
        Which model is on bottom in the ratio.

    Output
    ------
    ratio: pd.DataFrame
    """
    avg = results.groupby(['model', grp_var])[metric].mean().reset_index()

    ratios = []
    grp_var_values = np.unique(avg[grp_var])
    for val in grp_var_values:
        avg_this_val = avg.query('{} == @val'.format(grp_var))

        val_top = avg_this_val.query("model == @model_top")[metric].item()
        val_bot = avg_this_val.query("model == @model_bot")[metric].item()

        diff = val_top - val_bot
        if abs(val_bot) < np.finfo(float).eps:
            ratio = np.inf
        else:
            ratio = val_top / val_bot

        ratios.append({grp_var: val,
                       '{}__ratio'.format(metric): ratio,
                       '{}__diff'.format(metric): diff})

    ratios = pd.DataFrame(ratios)
    return ratios


def std_root_n(x):
    """
    Returns the standard deviation divided by the square root of the number of samples
    """
    return np.std(x) / np.sqrt(len(x))
