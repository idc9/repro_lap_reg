import numpy as np
from sklearn.utils import check_random_state
from scipy.linalg import block_diag

from fclsp.reshaping_utils import fill_hollow_sym


def complete_graph(n_nodes=100, edges='ones', pos=True, random_state=None):
    """

    Parameters
    ----------
    n_nodes: int

    edges: str
        How the edges are set. Must be one of ['ones', 'normal', 'uniform'].
        'ones': edges are symmetric Bernoulli random variables. If pos=True, then all edges are one.

        'normal': edges are N(0, 1) or |N(0, 1)| depending on pos

        'uniform': edges U(-1, 1) or U(0, 1) depending on pos

    pos: bool
        Make edges positive.

    random_state: None, int

    Output
    ------
    A: array-like, (n_nodes, n_nodes)
    """
    rng = check_random_state(random_state)

    assert edges in ['ones', 'normal', 'uniform']

    n_edges = int(n_nodes * (n_nodes - 1) / 2)

    if edges == 'ones':
        if pos:
            a = np.ones(n_edges)
        else:
            a = rng.choice([-1, 1], size=n_edges, p=[.5, .5])

    elif edges == 'normal':
        a = rng.normal(size=n_edges)

        if pos:
            a = np.abs(a)

    elif edges == 'uniform':
        a = rng.uniform(low=-1, high=1, size=n_edges)

        if pos:
            a = np.abs(a)

    A = fill_hollow_sym(a)

    return A


def block_diag_complete(block_sizes=[10, 20, 30],
                        edges='ones', pos=True, random_state=None):

    rng = check_random_state(random_state)

    return block_diag(*[complete_graph(n_nodes=s, edges=edges,
                                       pos=pos, random_state=rng)
                        for s in block_sizes])
