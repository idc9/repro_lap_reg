import numpy as np
from copy import deepcopy
import networkx as nx
from itertools import combinations
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.cluster import spectral_clustering

import warnings

from fclsp.reshaping_utils import vec_hollow_sym


def compare_vecs(est, truth, zero_tol=0):
    """

    Parameters
    ----------
    est: array-like
        The estimated vector.

    truth: array-like
        The true vector parameter.

    zero_tol: float
        Zero tolerance for declaring an element equal to zero.

    Output
    ------
    out: dict
        Dictonary containing various error measurements.
    """
    est = np.array(est).reshape(-1)
    truth = np.array(truth).reshape(-1)

    # true norms
    true_L2 = np.sqrt((truth ** 2).sum())
    true_L1 = abs(truth).sum()

    # we will divide by this later so make sure it isn't zero
    true_L2 = max(true_L2, np.finfo(float).eps)
    true_L1 = max(true_L1, np.finfo(float).eps)

    assert len(est) == len(truth)

    support_est = abs(est) > zero_tol
    support_true = abs(truth) > zero_tol

    n = len(est)
    resid = est - truth
    out = {}

    ####################
    # size of residual #
    ####################
    out['L2'] = np.sqrt((resid ** 2).sum())
    out['L1'] = abs(resid).sum()
    out['L2_rel'] = out['L2'] / true_L2
    out['L1_rel'] = out['L1'] / true_L1
    out['MSE'] = out['L2'] / np.sqrt(n)
    out['MAE'] = out['L1'] / n
    out['max'] = abs(resid).max()

    # compare supports
    out.update(compare_supports(support_est, support_true))

    out['support_auc'] = roc_auc_score(y_true=support_true,
                                       y_score=abs(est))

    #################
    # compare signs #
    #################

    _est = deepcopy(est)
    _est[~support_est] = 0

    _truth = deepcopy(truth)
    _truth[~support_true] = 0

    out['sign_error'] = np.mean(np.sign(_est) != np.sign(_truth))
    # # only compute at true non-zero
    # est_at_true_nz = est[nz_mask_true]
    # true_at_true_nz = truth[nz_mask_true]
    # sign_misses = np.sign(est_at_true_nz) != np.sign(true_at_true_nz)

    # out['sign_error'] = np.mean(sign_misses)

    return out


def compare_adj_mats(est, truth, zero_tol=0):
    """
    Compares the graph structure of two estimated adjacency matrix structure parameters. This ignores the diagonal entries and binariezes the off diagonal entries to be the T/F support mask.

    Parameteres
    -----------
    est: arary-like, shape (n_nodes, n_nodes)

    truth: arary-like, shape (n_nodes, n_nodes)

    zero_tol: float
        Entries in absolute values smaller than this are killed.

    Ouput
    -----
    out: dict of floats
    """

    assert (est.shape[0] == truth.shape[0]) and \
        (est.shape[1] == truth.shape[1])

    assert est.shape[0] == est.shape[1]  # make sure symmetric

    # get adjacency matrices i.e. binarize edges and
    # make sure diagonal elements are zero
    A_est = np.array(deepcopy(est))
    A_est = abs(A_est) > zero_tol
    np.fill_diagonal(A_est, 0)

    A_truth = np.array(deepcopy(truth))
    A_truth = abs(A_truth) > zero_tol
    np.fill_diagonal(A_truth, 0)

    # compute all sorts of graph metrics for the binary graphs
    out = compare_graphs(est=A_est, truth=A_truth)

    # run spectral clustering on the estimated adjacency matrix with weighted edges
    A_est = np.array(deepcopy(est))
    np.fill_diagonal(A_est, 0)
    A_est = np.abs(A_est)

    try:
        # this sometimes crashes for reasons I have yet to determine so we put it in a try-catch statement

        # TODO: handle graphs with multiple connected components better!
        warnings.simplefilter("ignore")
        mem_est = spectral_clustering(A_est, n_clusters=out['n_blocks_true'])

        mem_true = get_block_memberships(truth)  # true memberships

        out['block_ars_weighted_adj'] = \
            adjusted_rand_score(labels_true=mem_true,
                                labels_pred=mem_est)

    except Exception as e:
        print("spectral_clustering failed in compare_adj_mats")
        print(e)
        print(A_est.shape)
        print(A_est)
        out['block_ars_weighted_adj'] = np.nan

    return out


def compare_graphs(est, truth):
    """
    Parameters
    ----------
    est: array-like, shape (n_nodes, n_nodes)

    true: array-like, shape (n_nodes, n_nodes)

    Output
    ------
    dict of floats
    """
    out = {}

    # get the blocks of the estiamted and true matrices
    blocks_est = get_blocks(est)
    blocks_true = get_blocks(truth)

    n_blocks_true = len(blocks_true)

    # fill in within block edges
    BS_est = get_block_support(blocks_est)
    BS_true = get_block_support(blocks_true)

    # block memberships
    mem_true = get_block_memberships(truth)
    mem_est = get_block_memberships(est)

    BS_est = vec_hollow_sym(BS_est)
    BS_true = vec_hollow_sym(BS_true)

    try:
        # this sometimes crashes for reasons I have yet to determine so we put it in a try-catch statement

        # run spectral clustering on est adjaceny matrix
        # where we know the true nubmer of blocks
        mem_spect_clust_est = spectral_clustering(est.astype(int),
                                                  n_clusters=n_blocks_true)

        out['block_ars_spect_clust'] = \
            adjusted_rand_score(labels_true=mem_true,
                                labels_pred=mem_spect_clust_est)

    except Exception as e:
        print("spectral_clustering failed in compare_graphs")
        print(e)
        print(est.shape)
        print(est.astype(int))

        out['block_ars_spect_clust'] = np.nan

    out['n_blocks_est'] = len(blocks_est)
    out['n_blocks_true'] = n_blocks_true

    out['block_sizes_est'] = [len(b) for b in blocks_est]
    out['block_sizes_true'] = [len(b) for b in blocks_true]

    # Adjusted rand score for block memberships
    out['block_ars'] = adjusted_rand_score(labels_true=mem_true,
                                           labels_pred=mem_est)

    # add support metrics for filled in graphs
    bs_out = compare_supports(supp_est=BS_est, supp_true=BS_true)
    for k in bs_out.keys():
        out['block_supp_{}'.format(k)] = bs_out[k]

    return out


def compare_supports(supp_est, supp_true):
    """
    Computes various metrics for estimating the support of a parameter vector.
    We say postive = non-zero element of true parameter
    (so negative = numer of zeros in true parameters)

    Parameters
    ----------
    supp_est: array-like of bools
        Estimated support; True = non-zero.

    supp_true: array-like of bools
        Support of true vector; True = non-zero.

    Output
    ------
    out: dict
    """
    supp_est = np.array(supp_est).reshape(-1).astype(bool)
    supp_true = np.array(supp_true).reshape(-1).astype(bool)
    assert len(supp_est) == len(supp_true)

    P = sum(supp_true)  # number of poistives (= non zero elts) in true parameter
    N = sum(~supp_true)  # number of negatives (= zero elts) in true parameter

    TP = sum(supp_true & supp_est)
    TN = sum((~supp_true) & (~supp_est))
    FP = sum((~supp_true) & supp_est)
    FN = sum(supp_true & (~supp_est))

    out = {}
    out['n_nonzero_est'] = sum(supp_est)
    out['n_nonzero_true'] = sum(supp_true)

    out['n_true_pos'] = TP
    out['n_true_neg'] = TN
    out['n_false_pos'] = FP
    out['n_false_neg'] = FN

    # true positive rates
    if P > 0:
        out['true_pos_rate'] = TP / P
        out['false_neg_rate'] = 1 - out['true_pos_rate']
    else:
        out['true_pos_rate'] = np.nan
        out['false_neg_rate'] = np.nan

    # true negative rates
    if N > 0:
        out['true_neg_rate'] = TN / N
        out['false_pos_rate'] = 1 - out['true_neg_rate']
    else:
        out['true_neg_rate'] = np.nan
        out['false_pos_rate'] = np.nan

    out['support_error'] = np.mean(supp_est != supp_true)

    return out


def get_blocks(A):
    """
    Gets the blocks of a block diagonal matrix.

    Parameters
    ----------
    A: array-like, (n_nodes, n_nodes)
        The adjacency matrix.

    Output
    ------
    blocks: list of list of ints
        Each entry of blocks is a list containing the indices of a block.
    """
    G = nx.from_numpy_array(A)
    cc_nodes = [list(cc) for cc in nx.connected_components(G)]
    return cc_nodes


def get_block_support(blocks):
    """
    Parameters
    ----------
    blocks: list of list of ints
        Each entry of blocks is a list containing the indices of a block.
        This is the output of get_blocks()

    Output
    ------
    BS: array-like, shape (n_nodes, n_nodes)
        Block support adjacency matrix.


    """
    n_nodes = sum(len(b) for b in blocks)

    # print('\n\n')
    # print(blocks)
    # print([len(b) for b in blocks])
    # print(n_nodes)

    BS = np.zeros((n_nodes, n_nodes))
    for idxs in blocks:
        for (i, j) in combinations(idxs, 2):
            BS[i, j] = 1
            BS[j, i] = 1

    return BS


def get_block_memberships(A):
    """
    Gets the block membership each node in a block diagonal matrix.

    Parameters
    ----------
    A: array-like, (n_nodes, n_nodes)
        The adjacency matrix.

    Output
    ------
    memberships: array-like, shape (n_nodes, )
        Vector denoting the block membership of each node.
    """
    blocks = get_blocks(A)
    n_nodes = sum(len(b) for b in blocks)

    memberships = np.zeros(n_nodes)
    for block_idx, nodes in enumerate(blocks):
        for node in nodes:
            memberships[node] = block_idx

    return memberships
