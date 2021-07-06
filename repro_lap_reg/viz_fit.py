import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from sklearn.linear_model import LassoCV

from repro_lap_reg.viz_utils import maybe_save

from ya_glm.cv.cv_viz import plot_cv_path


def viz_path_diagnostics(out, models, model_name, log_param=True,
                         save_dir=None, inches=5):

    save = lambda name: maybe_save(name=name,  # 'vs_{}_{}'.format(vs, name),
                                   save_dir=save_dir,
                                   dpi=100, close=True)

    figsize = (inches, inches)
    # cross-validation train/test score paths
    if models is not None:
        est = models['fit'][model_name]

        #########################
        # cross-validation path #
        #########################

        plt.figure(figsize=figsize)
        if 'mean_test_L2' in \
                est.cv_results_.keys():
            score_name = 'L2'
        else:
            score_name = 'score'

        # negate means est and covar scores
        negate = not hasattr(est.best_estimator_, 'coef_')

        plot_cv_path(est.cv_results_,
                     metric=score_name,
                     param='pen_val',
                     log_param=log_param,
                     negate=negate,
                     selected_idx=est.best_tune_idx_)

        # for opt history below
        best_est = est.best_estimator_

        plt.title('Cross-validation score path')
        save('cv_tr_tst_score_path.png')

        ############################
        # LLA optimization history #
        ############################

        if hasattr(best_est, 'solve_lla'):

            plt.figure(figsize=(2 * inches, inches))
            plot_laa_loss(best_est.opt_data_,
                          plot_diff=True)  # , log_diff=True

            plt.title('LLA optimization history')
            save('lla_opt_history.png')

    # # pull out information we need
    # path_results = out['path']['results'].\
    #     query("model == @model_name & vs == '@vs'")

    param_seq = out['path']['param_seq'][model_name]
    best_idx = out['path']['best_idx'][model_name]

    # format keyword args for path plotting
    kws = {#'path_results': path_results,
           'param_seq': param_seq,
           'best_idx': best_idx,
           'palette': 'Set2',
           'figsize': figsize,
           'log_param': log_param
           }

    ####################
    # vs. truth/oracle #
    ####################
    for vs in ['truth', 'oracle']:
        # pull out information we need
        path_results = out['path']['results'].\
            query("model == @model_name & vs == @vs")

        # true path, est vs true L2 error
        plot_path(path_results=path_results, metrics='L2', **kws)
        plt.title('Estimate vs. {}'.format(vs))
        plt.ylim(0)
        save('vs_{}_path_L2_error.png'.format(vs))

        # true path, est vs true support error
        plot_path(path_results=path_results,
                  metrics='support_error', **kws)
        plt.ylim(0, 1)
        plt.title('Estimate vs. {}'.format(vs))
        save('vs_{}_path_support_error.png'.format(vs))

        # true path, est vs true TN/TP rates
        plot_path(path_results=path_results,
                  metrics=['true_neg_rate', 'true_pos_rate'], **kws)
        plt.ylim(0, 1)
        plt.title('Estimate vs. {}'.format(vs))
        save('vs_{}_path_tnr_tpr.png'.format(vs))

    ############
    # support #
    ############

    # pull out information we need
    path_results = out['path']['results'].\
        query("model == @model_name & vs == 'truth'")

    # Estimated support
    true_nonzero = out['path']['results']['n_nonzero_true'].values[0]
    plot_path(path_results=path_results, metrics='n_nonzero_est', **kws)
    plt.axhline(true_nonzero, color='red', label='true support {}'.
                                                 format(true_nonzero))
    plt.title('Estimated support size')
    save('path_est_support.png')

    # Estimated support
    true_n_blocks = out['path']['results']['n_blocks_true'].values[0]
    plot_path(path_results=path_results, metrics='n_blocks_est', **kws)
    plt.axhline(true_n_blocks, color='red', label='true n_block {}'.
                                                  format(true_n_blocks))
    plt.legend()
    plt.title('Estimated number of blocks')
    save('path_est_n_blocks.png')

    # true path, est vs true support error
    plot_path(path_results=path_results, metrics='runtime', **kws)
    plt.title('Path runtimes')
    plt.title('Runtime (s)')
    save('path_runtimes.png')


#################
# array heatmap #
#################


def heatmap(X, zero_thresh=1e-6, show_support=False):
    """
    Plots the heatmap of a matrix X.

    Parameters
    -----------
    X: array-like

    zero_thresh: float
        Entries at or below this threshold are declared zero.

    show_support: bool
        If True will plot the suppor matrix.
    """
    X = np.array(X)
    assert X.ndim == 2
    supp = abs(X) > zero_thresh

    if show_support:
        plt.subplot(1, 2, 1)

    sns.heatmap(X, mask=~supp, center=0, cmap='RdBu')

    if show_support:
        plt.subplot(1, 2, 2)

        sns.heatmap(supp, cmap='Blues', cbar=False)

#############
# true path #
#############


def plot_path(path_results, metrics, param_seq,
              best_idx=None, palette='Set2',
              figsize=None, log_param=False):
    """
    Plots a metric for values along a path.

    Parameters
    ----------
    metric: str or list of str

    path_error: list of dicts

    param_seq: array-like

    param_name: str, None

    best_idx: int, None

    """
    if figsize is not None:
        plt.figure(figsize=figsize)

    if isinstance(metrics, str):
        metrics = [metrics]

    n_metrics = len(metrics)
    colors = sns.color_palette(palette=palette, n_colors=n_metrics)

    label = None

    if log_param:
        param_seq = np.log10(param_seq)

    # plot each metric
    for i in range(n_metrics):
        if n_metrics > 1:
            label = metrics[i]

        plt.plot(param_seq, path_results[metrics[i]],
                 color=colors[i], marker='.', label=label)

    # only label if one metric
    if n_metrics == 1:
        plt.ylabel(metrics[0])

    if log_param:
        plt.xlabel('log10({})'.format(param_seq.name))
    else:
        plt.xlabel(param_seq.name)

    if best_idx is not None:

        if log_param:
            label = 'log10(best value) = '
        else:
            label = 'best value = '

        label += '{:1.4f}'.format(param_seq[best_idx])

        plt.axvline(param_seq[best_idx], color='black', ls='--',
                    label=label)

    if n_metrics > 1 or best_idx is not None:
        plt.legend()


def print_estimators(estimators, print_func=print):
    """
    Printout a collection of estimators.

    Parameters
    ----------
    estimators: dict
        The estimators to print.

    print_func: callable
        The function used to print.
    """

    names = np.sort(list(estimators.keys()))

    for name in names:

        est = estimators[name]
        print_func(name)
        print_func(est)
        print_func(est.get_params(deep=True))

        if hasattr(est, 'best_estimator_') and est.best_estimator_ is not None:
            print_func('\n{}, best estimator'.format(name))
            print_func(est.best_estimator_)
            print_func(est.best_estimator_.get_params(deep=True))

        if hasattr(est, 'init_est_') and est.init_est_ is not None:
            print_func('\n{}, initial estimator'.format(name))
            print_func(est.init_est_)
            print_func(est.init_est_.get_params(deep=True))

        print_func('\n\n\n\n{}\n\n\n\n'.format('-' * 50))

########################
# optimization history #
########################


def plot_laa_loss(lla_opt_data, plot_diff=True):  # , log_diff=True

    loss_vals = lla_opt_data['obj']

    if plot_diff:
        plt.subplot(1, 2, 1)

    plt.plot(loss_vals, marker='.')
    plt.xlabel("LLA step")
    plt.ylabel("Loss function")

    if plot_diff:
        plt.subplot(1, 2, 2)

        diffs = abs(np.diff(loss_vals))

        # if log_diff:

        #     diffs = np.log10(diffs[diffs > np.finfo(float).eps])
        #     ylabel = 'log10(difference)'
        # else:
        ylabel = 'difference'

        plt.plot(diffs, marker='.')
        plt.xlabel("LLA step")
        plt.ylabel(ylabel)
